import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// Segmentador U-Net corriendo en un Isolate secundario.
///
/// Motivo del cambio:
///   La versión anterior corría en el hilo principal (UI thread), lo que
///   bloqueaba la interfaz durante la inferencia (~200-500ms).
///   Al mover la inferencia a un Isolate, el UI permanece fluido mientras
///   el modelo procesa la imagen en segundo plano.
///
/// Protocolo de mensajes:
///   init   → [modelBytes, replyPort] → 'init_success' | ['init_error', msg]
///   segment → [flatInput, replyPort] → ['success', Float32List] | ['error', msg]
///   close  → termina el Isolate
class UnetSegmenterInterpreter {
  SendPort? _isolateSendPort;
  Isolate? _isolate;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  /// Carga el modelo U-Net en un Isolate secundario de forma asíncrona.
  Future<void> initModel() async {
    if (_isLoaded) return;

    try {
      // 1. Leer el modelo desde los assets en el hilo principal
      final byteData = await rootBundle.load('assets/models/fused_segmenter_unet.onnx');
      final modelBytes = byteData.buffer.asUint8List();

      // 2. Lanzar el Isolate secundario
      final initPort = ReceivePort();
      _isolate = await Isolate.spawn(
        _unetIsolateEntryPoint,
        initPort.sendPort,
      );

      // 3. Recibir el SendPort del Isolate
      final childSendPort = await initPort.first as SendPort;
      _isolateSendPort = childSendPort;

      // 4. Enviar los bytes del modelo para inicialización
      final handshakePort = ReceivePort();
      _isolateSendPort!.send(['init', modelBytes, handshakePort.sendPort]);

      final initResult = await handshakePort.first;
      if (initResult is List && initResult[0] == 'init_success') {
        _isLoaded = true;
        print('✅ U-Net ONNX cargado exitosamente en Isolate de fondo.');
      } else {
        final errMsg = initResult is List ? initResult[1] : 'Error desconocido';
        throw Exception('Fallo al inicializar U-Net en Isolate: $errMsg');
      }
    } catch (e) {
      print('❌ Error cargando U-Net ONNX offline: $e');
      _isLoaded = false;
      close();
    }
  }

  /// Ejecuta la segmentación sobre una imagen aplanada [0, 1] de tamaño 49152 (128×128×3).
  /// Retorna un mapa probabilístico de 16384 valores (128×128×1).
  Future<Float32List?> segment(Float32List flatInputImage) async {
    if (!_isLoaded || _isolateSendPort == null) {
      print('❌ Error: El intérprete U-Net no está inicializado.');
      return null;
    }

    if (flatInputImage.length != 49152) {
      print(
        '❌ Error de dimensiones U-Net: '
        'Se esperaban 49152 (128×128×3), se obtuvieron ${flatInputImage.length}.',
      );
      return null;
    }

    try {
      final replyPort = ReceivePort();
      _isolateSendPort!.send(['segment', flatInputImage, replyPort.sendPort]);

      final response = await replyPort.first;
      if (response is List && response[0] == 'success') {
        return response[1] as Float32List;
      } else {
        final error = response is List ? response[1] : 'Segmentación fallida';
        print('❌ Error de segmentación en Isolate: $error');
        return null;
      }
    } catch (e) {
      print('❌ Error durante la inferencia U-Net: $e');
      return null;
    }
  }

  /// Libera la sesión de la U-Net terminando el Isolate.
  void close() {
    _isolateSendPort?.send('close');
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _isolateSendPort = null;
    _isLoaded = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ISOLATE ENTRY POINT — Corre en un hilo separado del UI
// ─────────────────────────────────────────────────────────────────────────────
void _unetIsolateEntryPoint(SendPort mainSendPort) {
  final childReceivePort = ReceivePort();
  mainSendPort.send(childReceivePort.sendPort);

  OrtSession? session;

  childReceivePort.listen((message) {
    if (message is List && message[0] == 'init') {
      final Uint8List modelBytes = message[1];
      final SendPort replyPort = message[2];
      try {
        OrtEnv.instance; // inicializa el entorno ONNX Runtime
        final sessionOptions = OrtSessionOptions();
        session = OrtSession.fromBuffer(modelBytes, sessionOptions);
        replyPort.send(['init_success']);
      } catch (e) {
        replyPort.send(['init_error', e.toString()]);
      }
    } else if (message is List && message[0] == 'segment') {
      final Float32List flatInputImage = message[1];
      final SendPort replyPort = message[2];

      if (session == null) {
        replyPort.send(['error', 'Sesión U-Net no inicializada']);
        return;
      }

      try {
        // Tensor de entrada: [1, 3, 128, 128] (formato NCHW)
        final inputShape = [1, 3, 128, 128];
        final inputTensor = OrtValueTensor.createTensorWithDataList(
          flatInputImage,
          inputShape,
        );

        final inputs = {'image_input': inputTensor};
        final runOptions = OrtRunOptions();
        final outputs = session!.run(runOptions, inputs);

        if (outputs.isEmpty) {
          replyPort.send(['error', 'U-Net retornó resultado vacío.']);
          inputTensor.release();
          return;
        }

        // Salida: [1, 1, 128, 128]
        final rawOutput = outputs[0]?.value;
        final flatMask = Float32List(128 * 128);
        int idx = 0;

        void flatten(dynamic item) {
          if (item is List) {
            for (var i = 0; i < item.length; i++) {
              flatten(item[i]);
            }
          } else if (item is num) {
            if (idx < 16384) {
              flatMask[idx++] = item.toDouble();
            }
          }
        }

        if (rawOutput != null) {
          flatten(rawOutput);
        }

        if (idx != 16384) {
          replyPort.send(['error', 'Dimensión de salida U-Net incorrecta: se esperaban 16384 elementos, se obtuvieron \$idx.']);
          inputTensor.release();
          for (var element in outputs) {
            element?.release();
          }
          return;
        }

        replyPort.send(['success', flatMask]);

        // Liberar recursos nativos
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
