import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'ui/image_acquisition_screen.dart';
import 'model_runner/unet_segmenter_interpreter.dart';
import 'model_runner/fused_model_interpreter.dart';

// Providers globales para gestionar el estado de los dos modelos de IA offline
final unetProvider = Provider((ref) => UnetSegmenterInterpreter());
final classifierProvider = Provider((ref) => FusedModelInterpreter());

void main() async {
  // Asegurar que las integraciones nativas de Flutter estén inicializadas
  WidgetsFlutterBinding.ensureInitialized();

  // Los modelos se inicializan bajo demanda a través de los Riverpod Providers
  // (unetProvider y classifierProvider) cuando la pantalla de resultados los solicita.

  runApp(
    const ProviderScope(
      child: DermaScanApp(),
    ),
  );
}

class DermaScanApp extends StatelessWidget {
  const DermaScanApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DermaScan AI',
      debugShowCheckedModeBanner: false,
      // Aplicar tema clínico oscuro premium con la tipografía Google Font "Outfit"
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0F0F1A),
        primaryColor: const Color(0xFF00FFCC),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00FFCC),
          secondary: Color(0xFF00B3FF),
          background: Color(0xFF0F0F1A),
          surface: Color(0xFF1E1E2C),
        ),
        useMaterial3: true,
      ),
      home: const ImageAcquisitionScreen(),
    );
  }
}
