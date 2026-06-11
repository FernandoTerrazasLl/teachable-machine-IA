// Test unitario de inicialización de widgets para DermaScan AI.
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:derma_scan_app/main.dart';

void main() {
  testWidgets('App initialization smoke test', (WidgetTester tester) async {
    // Construir la aplicación dentro del ámbito del proveedor requerido por Riverpod.
    await tester.pumpWidget(
      const ProviderScope(
        child: DermaScanApp(),
      ),
    );

    // Verificar que la pantalla inicial de adquisición de imágenes se cargue.
    expect(find.textContaining('DermaScan'), findsOneWidget);
  });
}
