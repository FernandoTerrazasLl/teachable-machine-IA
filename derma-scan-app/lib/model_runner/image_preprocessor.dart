import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

class ImagePreprocessor {
  static Future<Uint8List> getResizedPixels(File imageFile, int targetWidth, int targetHeight) async {
    final Uint8List imageBytes = await imageFile.readAsBytes();
    final ui.Codec codec = await ui.instantiateImageCodec(
      imageBytes,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
    );
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    final ui.Image image = frameInfo.image;
    
    final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw Exception("Fallo al extraer píxeles de la imagen.");
    }
    return byteData.buffer.asUint8List();
  }

  static Float32List preprocessForUnet(Uint8List rgbaBytes) {
    final int size = 128 * 128;
    final Float32List processed = Float32List(size * 3);

    for (int i = 0; i < size; i++) {
      final int pixelOffset = i * 4;
      final double r = rgbaBytes[pixelOffset] / 255.0;
      final double g = rgbaBytes[pixelOffset + 1] / 255.0;
      final double b = rgbaBytes[pixelOffset + 2] / 255.0;

      processed[i] = r;
      processed[i + size] = g;
      processed[i + 2 * size] = b;
    }

    return processed;
  }

  static Float32List preprocessAndApplyMask(
    Uint8List rgba260,
    Float32List unetMask128,
  ) {
    final int size260 = 260 * 260;
    final Float32List processed = Float32List(size260 * 3);

    for (int y = 0; y < 260; y++) {
      for (int x = 0; x < 260; x++) {
        final int index260 = y * 260 + x;
        final int pixelOffset = index260 * 4;

        final int maskX = (x * 128 / 260).floor().clamp(0, 127);
        final int maskY = (y * 128 / 260).floor().clamp(0, 127);
        final double maskValue = unetMask128[maskY * 128 + maskX];

        final double weight = maskValue >= 0.35 ? 1.0 : 0.0;

        final double r = (rgba260[pixelOffset] / 255.0) * weight;
        final double g = (rgba260[pixelOffset + 1] / 255.0) * weight;
        final double b = (rgba260[pixelOffset + 2] / 255.0) * weight;

        processed[index260] = r;
        processed[index260 + size260] = g;
        processed[index260 + 2 * size260] = b;
      }
    }

    return processed;
  }

  static Uint8List getMaskedRGBA(Uint8List rgba260, Float32List unetMask128) {
    final int size260 = 260 * 260;
    final Uint8List masked = Uint8List(size260 * 4);

    for (int y = 0; y < 260; y++) {
      for (int x = 0; x < 260; x++) {
        final int index260 = y * 260 + x;
        final int pixelOffset = index260 * 4;

        final int maskX = (x * 128 / 260).floor().clamp(0, 127);
        final int maskY = (y * 128 / 260).floor().clamp(0, 127);
        final double maskValue = unetMask128[maskY * 128 + maskX];

        final double weight = maskValue >= 0.35 ? 1.0 : 0.0;

        masked[pixelOffset] = (rgba260[pixelOffset] * weight).round().clamp(0, 255);
        masked[pixelOffset + 1] = (rgba260[pixelOffset + 1] * weight).round().clamp(0, 255);
        masked[pixelOffset + 2] = (rgba260[pixelOffset + 2] * weight).round().clamp(0, 255);
        masked[pixelOffset + 3] = 255;
      }
    }
    return masked;
  }

  static Future<Uint8List> getMaskedPngBytes(Uint8List rgba260, Float32List unetMask128) async {
    final Uint8List maskedRgba = getMaskedRGBA(rgba260, unetMask128);
    final Completer<ui.Image> completer = Completer();
    ui.decodeImageFromPixels(
      maskedRgba,
      260,
      260,
      ui.PixelFormat.rgba8888,
      (ui.Image img) {
        completer.complete(img);
      },
    );
    final ui.Image image = await completer.future;
    final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) {
      throw Exception("Error al codificar la imagen segmentada.");
    }
    return byteData.buffer.asUint8List();
  }

  static Float32List applyMorphologicalClosing(Float32List mask, double threshold) {
    const int width = 128;
    const int height = 128;
    final Uint8List binary = Uint8List(width * height);
    
    for (int i = 0; i < mask.length; i++) {
      binary[i] = mask[i] >= threshold ? 1 : 0;
    }
    
    final Uint8List dilated = Uint8List(width * height);
    const int r = 2; 
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int maxVal = 0;
        for (int dy = -r; dy <= r; dy++) {
          final int ny = y + dy;
          if (ny < 0 || ny >= height) continue;
          for (int dx = -r; dx <= r; dx++) {
            final int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            if (binary[ny * width + nx] == 1) {
              maxVal = 1;
              break;
            }
          }
          if (maxVal == 1) break;
        }
        dilated[y * width + x] = maxVal;
      }
    }
    
    final Float32List closed = Float32List(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int minVal = 1;
        for (int dy = -r; dy <= r; dy++) {
          final int ny = y + dy;
          if (ny < 0 || ny >= height) continue;
          for (int dx = -r; dx <= r; dx++) {
            final int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            if (dilated[ny * width + nx] == 0) {
              minVal = 0;
              break;
            }
          }
          if (minVal == 0) break;
        }
        closed[y * width + x] = minVal.toDouble();
      }
    }
    
    return closed;
  }

  static double calculateAverageBrightness(Uint8List rgbaBytes) {
    double sum = 0.0;
    final int size = rgbaBytes.length ~/ 4;
    for (int i = 0; i < size; i++) {
      final int offset = i * 4;
      sum += rgbaBytes[offset] + rgbaBytes[offset + 1] + rgbaBytes[offset + 2];
    }
    return sum / (size * 3);
  }

  static int countActiveCentralPixels(Float32List mask128) {
    int count = 0;
    for (int y = 32; y < 96; y++) {
      for (int x = 32; x < 96; x++) {
        if (mask128[y * 128 + x] >= 0.5) {
          count++;
        }
      }
    }
    return count;
  }
}
