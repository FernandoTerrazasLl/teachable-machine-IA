import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class FusedModelInterpreter {
  SendPort? _isolateSendPort;
  Isolate? _isolate;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  /// Carga el modelo fusionado .onnx en un Isolate secundario de forma asíncrona
  Future<void> initModel() async {
    if (_isLoaded) return;

    try {
      // 1. Leer el archivo del modelo desde los assets de Flutter en el hilo principal
      final byteData = await rootBundle.load('assets/models/fused_dermatology_model.onnx');
      final modelBytes = byteData.buffer.asUint8List();

      // 2. Crear el puerto de comunicación y levantar el Isolate secundario
      final initPort = ReceivePort();
      _isolate = await Isolate.spawn(_fusedModelIsolateEntryPoint, initPort.sendPort);

      // Esperar a recibir el SendPort del Isolate secundario
      final childSendPort = await initPort.first as SendPort;
      _isolateSendPort = childSendPort;

      // 3. Inicializar el modelo enviando los bytes y un puerto para confirmación
      final handshakePort = ReceivePort();
      _isolateSendPort!.send(['init', modelBytes, handshakePort.sendPort]);
      
      final initResult = await handshakePort.first;
      if (initResult is List && initResult[0] == 'init_success') {
        _isLoaded = true;
        print("✅ Modelo unificado ONNX cargado exitosamente offline en Isolate de fondo.");
      } else {
        final errMsg = initResult is List ? initResult[1] : 'Error desconocido';
        throw Exception("Fallo al inicializar modelo en Isolate: $errMsg");
      }
    } catch (e) {
      print("❌ Error cargando modelo ONNX offline: $e");
      _isLoaded = false;
      close();
    }
  }

  /// Ejecuta la inferencia sobre una lista de píxeles aplanados en rango [0, 1]
  /// Recibe un Float32List de tamaño 202800 (260 x 260 x 3 — EfficientNet-B2)
  /// Retorna un mapa con las probabilidades de AK, BCC y MEL
  Future<Map<String, double>?> predict(Float32List flatPixels) async {
    if (!_isLoaded || _isolateSendPort == null) {
      print("❌ Error: El intérprete ONNX no está inicializado.");
      return null;
    }

    if (flatPixels.length != 202800) {
      print("❌ Error de dimensiones: Se esperaban 202800 píxeles (260x260x3), se obtuvieron ${flatPixels.length}.");
      return null;
    }

    try {
      final replyPort = ReceivePort();
      _isolateSendPort!.send(['predict', flatPixels, replyPort.sendPort]);
      
      final response = await replyPort.first;
      if (response is List && response[0] == 'success') {
        return Map<String, double>.from(response[1]);
      } else {
        final error = response is List ? response[1] : 'Inferencia fallida';
        print("❌ Error de predicción en Isolate: $error");
        return null;
      }
    } catch (e) {
      print("❌ Error durante la inferencia ONNX: $e");
      return null;
    }
  }

  /// Libera la sesión de ONNX Runtime terminando el Isolate
  void close() {
    _isolateSendPort?.send('close');
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _isolateSendPort = null;
    _isLoaded = false;
  }
}

/// Punto de entrada del Isolate secundario que corre de forma asíncrona
void _fusedModelIsolateEntryPoint(SendPort mainSendPort) {
  final childReceivePort = ReceivePort();
  mainSendPort.send(childReceivePort.sendPort);

  OrtSession? session;

  childReceivePort.listen((message) {
    if (message is List && message[0] == 'init') {
      final Uint8List modelBytes = message[1];
      final SendPort replyPort = message[2];
      try {
        final env = OrtEnv.instance;
        final sessionOptions = OrtSessionOptions();
        session = OrtSession.fromBuffer(modelBytes, sessionOptions);
        replyPort.send(['init_success']);
      } catch (e) {
        replyPort.send(['init_error', e.toString()]);
      }
    } else if (message is List && message[0] == 'predict') {
      final Float32List flatPixels = message[1];
      final SendPort replyPort = message[2];
      
      if (session == null) {
        replyPort.send(['error', 'Sesión no inicializada']);
        return;
      }
      
      try {
        // 1. Crear el tensor de entrada de ONNX con dimensiones [1, 202800]
        final inputShape = [1, 202800];
        final inputTensor = OrtValueTensor.createTensorWithDataList(
          flatPixels,
          inputShape,
        );

        // 2. Definir los nombres de entrada y salida
        final inputs = {'raw_pixels': inputTensor};
        final runOptions = OrtRunOptions();

        // 3. Ejecutar inferencia offline en el hilo del Isolate
        final outputs = session!.run(runOptions, inputs);

        if (outputs.isEmpty) {
          replyPort.send(['error', 'Inferencia retornó un resultado vacío.']);
          inputTensor.release();
          return;
        }

        // 4. Leer las probabilidades de salida de forma segura
        final rawOutput = outputs[0]?.value;
        List<double> probabilities;
        if (rawOutput is List<List<double>>) {
          probabilities = rawOutput[0];
        } else if (rawOutput is List) {
          if (rawOutput.first is List) {
            probabilities = List<double>.from(rawOutput.first);
          } else {
            probabilities = List<double>.from(rawOutput);
          }
        } else {
          throw Exception("Formato de salida ONNX inesperado: ${rawOutput.runtimeType}");
        }

        // 5. Mapear a las clases oficiales del proyecto
        final result = {
          'AK': probabilities[0],  // Queratosis Actínica
          'BCC': probabilities[1], // Carcinoma Basocelular
          'MEL': probabilities[2], // Melanoma
        };

        replyPort.send(['success', result]);

        // 6. Liberar memoria nativa
        inputTensor.release();
        for (var element in outputs) {
          element?.release();
        }
      } catch (e) {
        replyPort.send(['error', e.toString()]);
      }
    } else if (message == 'close') {
      session?.release();
      Isolate.exit();
    }
  });
}
