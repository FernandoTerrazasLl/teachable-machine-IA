import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

import '../main.dart';
import '../model_runner/image_preprocessor.dart';

class ResultsScreen extends ConsumerStatefulWidget {
  final File imageFile;

  const ResultsScreen({Key? key, required this.imageFile}) : super(key: key);

  @override
  ConsumerState<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends ConsumerState<ResultsScreen> {
  bool _isLoading = true;
  String _loadingPhase = "Cargando imagen...";
  String? _errorMessage;
  
  Map<String, double>? _predictionResults;
  double _melanomaProb = 0.0;
  double _bccProb = 0.0;
  double _akProb = 0.0;
  Uint8List? _segmentedImageBytes;

  @override
  void initState() {
    super.initState();
    _runDiagnostics();
  }

  Future<void> _runDiagnostics() async {
    try {
      final unet = ref.read(unetProvider);
      final classifier = ref.read(classifierProvider);
      
      if (!unet.isLoaded || !classifier.isLoaded) {
        setState(() {
          _loadingPhase = "Inicializando motores de IA...";
        });
        await unet.initModel();
        await classifier.initModel();
      }

      setState(() {
        _loadingPhase = "Aislando la lesión sospechosa...";
      });
      final Uint8List rgba128 = await ImagePreprocessor.getResizedPixels(widget.imageFile, 128, 128);
      
      final double brightness = ImagePreprocessor.calculateAverageBrightness(rgba128);
      if (brightness < 15.0) {
        throw Exception("Mala iluminación detectada. Por favor, asegúrese de tomar la foto en un ambiente iluminado y enfoque la lesión.");
      }

      final Float32List inputUnet = ImagePreprocessor.preprocessForUnet(rgba128);

      final Float32List? mask128 = await unet.segment(inputUnet);
      if (mask128 == null) {
        throw Exception("Error al segmentar la lesión.");
      }

      final Float32List closedMask128 = ImagePreprocessor.applyMorphologicalClosing(mask128, 0.35);

      final int centralPixels = ImagePreprocessor.countActiveCentralPixels(closedMask128);
      if (centralPixels < 80) {
        throw Exception("Lesión no centrada o ausente. Por favor, coloque la lesión dentro del área central de la cámara.");
      }

      setState(() {
        _loadingPhase = "Analizando patrones patológicos...";
      });
      final Uint8List rgba260 = await ImagePreprocessor.getResizedPixels(widget.imageFile, 260, 260);
      final Float32List inputClassifier = ImagePreprocessor.preprocessAndApplyMask(rgba260, closedMask128);

      Uint8List? segmentedPng;
      try {
        segmentedPng = await ImagePreprocessor.getMaskedPngBytes(rgba260, closedMask128);
      } catch (_) {}

      final Map<String, double>? predictions = await classifier.predict(inputClassifier);
      if (predictions == null) {
        throw Exception("Error al clasificar la lesión.");
      }

      if (mounted) {
        setState(() {
          _predictionResults = predictions;
          _akProb = predictions['AK'] ?? 0.0;
          _bccProb = predictions['BCC'] ?? 0.0;
          _melanomaProb = predictions['MEL'] ?? 0.0;
          _segmentedImageBytes = segmentedPng;
          _isLoading = false;
        });
      }
    } catch (e) {

      if (mounted) {
        setState(() {
          _errorMessage = e.toString().replaceFirst("Exception: ", "");
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        title: const Text(
          "Análisis Clínico",
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: _isLoading
          ? _buildLoader()
          : _errorMessage != null
              ? _buildErrorView()
              : _buildResultsView(),
    );
  }

  /// Vista de Carga (Loader premium clínico)
  Widget _buildLoader() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SpinKitDoubleBounce(
            color: Color(0xFF00B3FF),
            size: 80.0,
          ),
          const SizedBox(height: 30),
          Text(
            _loadingPhase,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 16,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            "Esto se está calculando localmente en tu iPhone",
            style: TextStyle(color: Colors.white30, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorView() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent, size: 80),
            const SizedBox(height: 20),
            const Text(
              "Fallo en el Análisis",
              style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Text(
              _errorMessage ?? "Error desconocido en los modelos de IA.",
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 30),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.arrow_back),
              label: const Text("Volver a intentar"),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1E1E2C),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              ),
            )
          ],
        ),
      ),
    );
  }

  Widget _buildResultsView() {
    String maxClass = "MEL";
    double maxVal = _melanomaProb;
    if (_bccProb > maxVal) {
      maxClass = "BCC";
      maxVal = _bccProb;
    }
    if (_akProb > maxVal) {
      maxClass = "AK";
      maxVal = _akProb;
    }

    String classNameEs = "";
    String classDescription = "";
    Color severityColor = const Color(0xFF00B3FF);
    IconData severityIcon = Icons.check_circle_outline;

    switch (maxClass) {
      case "AK":
        classNameEs = "Queratosis Actínica (AK)";
        classDescription = "Se trata de una lesión pre-maligna superficial causada por la exposición solar prolongada. Aunque no es cáncer activo, requiere evaluación dermatológica para evitar su progresión.";
        severityColor = Colors.amber;
        severityIcon = Icons.warning_amber_rounded;
        break;
      case "BCC":
        classNameEs = "Carcinoma Basocelular (BCC)";
        classDescription = "Es el tipo de cáncer de piel más común. Es maligno pero de crecimiento lento y rara vez se disemina a otras partes del cuerpo. Debe ser evaluado por un dermatólogo para su extirpación.";
        severityColor = Colors.orange;
        severityIcon = Icons.warning_amber_rounded;
        break;
      case "MEL":
        classNameEs = "Sospecha de Melanoma (MEL)";
        classDescription = "Es el tipo de cáncer de piel más agresivo y requiere atención médica prioritaria. Detectado a tiempo tiene una tasa de curación muy alta mediante intervención quirúrgica.";
        severityColor = Colors.redAccent;
        severityIcon = Icons.gpp_maybe_outlined;
        break;
    }

    final bool isConclusive = maxVal >= 0.60;

    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Imagen Original",
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        height: 160,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white.withOpacity(0.1)),
                          image: DecorationImage(
                            image: FileImage(widget.imageFile),
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Segmentación por IA",
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 6),
                      _segmentedImageBytes != null
                          ? Container(
                              height: 160,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(color: Colors.white.withOpacity(0.1)),
                                image: DecorationImage(
                                  image: MemoryImage(_segmentedImageBytes!),
                                  fit: BoxFit.cover,
                                ),
                              ),
                            )
                          : Container(
                              height: 160,
                              decoration: BoxDecoration(
                                color: Colors.black,
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(color: Colors.white.withOpacity(0.1)),
                              ),
                              child: const Center(
                                child: Icon(Icons.blur_on, color: Colors.white30, size: 32),
                              ),
                            ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
 
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF1E1E2C),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: isConclusive 
                      ? const Color(0xFF00B3FF).withOpacity(0.4)
                      : Colors.amber.withOpacity(0.4),
                ),
              ),
              child: isConclusive
                  ? Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(severityIcon, color: severityColor, size: 28),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                classNameEs,
                                style: TextStyle(
                                  color: severityColor,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Text(
                          classDescription,
                          style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.4),
                        ),
                      ],
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.warning_amber_rounded, color: Colors.amber, size: 28),
                            SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                "Predicción No Concluyente",
                                style: TextStyle(
                                  color: Colors.amber,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Text(
                          "La confianza diagnóstica máxima obtenida es del ${(maxVal * 100).toStringAsFixed(1)}% (menor al umbral clínico requerido del 60.0%). Se sugiere realizar una valoración dermatológica manual directa.",
                          style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.4),
                        ),
                      ],
                    ),
            ),
            const SizedBox(height: 20),
 
            const Text(
              "Probabilidades Encontradas",
              style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 15),
            _buildProbabilityBar(
              label: "Melanoma (MEL) - Alta Gravedad",
              value: _melanomaProb,
              activeColor: Colors.redAccent,
            ),
            _buildProbabilityBar(
              label: "Carcinoma Basocelular (BCC) - Cáncer Común",
              value: _bccProb,
              activeColor: Colors.orange,
            ),
            _buildProbabilityBar(
              label: "Queratosis Actínica (AK) - Pre-cáncer",
              value: _akProb,
              activeColor: Colors.amber,
            ),
            const SizedBox(height: 20),
 
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.04),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: const Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.info_outline, color: Colors.white54, size: 20),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      "Este análisis es un resultado de cribado preliminar generado de forma automática por Inteligencia Artificial y no sustituye un diagnóstico clínico presencial por parte de un dermatólogo certificado.",
                      style: TextStyle(color: Colors.white54, fontSize: 11, height: 1.4),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
 
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00B3FF),
                foregroundColor: const Color(0xFF0F0F1A),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
              child: const Text(
                "Finalizar Diagnóstico",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildProbabilityBar({

    required String label,
    required double value,
    required Color activeColor,
  }) {
    final double pct = value * 100.0;
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
              Text(
                "${pct.toStringAsFixed(1)}%",
                style: TextStyle(color: activeColor, fontSize: 13, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value,
              minHeight: 8,
              backgroundColor: Colors.white.withOpacity(0.08),
              valueColor: AlwaysStoppedAnimation<Color>(activeColor),
            ),
          ),
        ],
      ),
    );
  }
}
