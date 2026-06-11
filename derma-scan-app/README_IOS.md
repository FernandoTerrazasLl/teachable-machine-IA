# 📱 Guía Oficial: Compilación e Instalación en iPhone 13 (iOS Offline)

Esta guía te guiará paso a paso para compilar e instalar la aplicación móvil offline **DermaScan AI** directamente en tu **iPhone 13** utilizando tu computadora Mac. 

Al estar en una Mac, tienes todas las herramientas nativas disponibles para firmar la aplicación e inyectar las librerías nativas de C++ OpenCV y ONNX Runtime.

---

## 📋 Requisitos Previos en tu Mac

1.  **Xcode:** Instalado desde la Mac App Store.
2.  **Flutter SDK:** Configurado y funcionando en tu sistema.
    *   Puedes verificarlo en la terminal ejecutando `flutter doctor`.
3.  **CocoaPods:** El gestor de dependencias de iOS. Si no lo tienes, instálalo ejecutando:
    ```bash
    sudo gem install cocoapods
    ```
4.  **Cuenta de Desarrollador de Apple:** Puedes usar tu cuenta gratuita de Apple ID (no es obligatorio pagar la suscripción anual de $99 USD para instalar la app en tu propio teléfono personal).

---

## 🛠️ Paso 1: Configurar el SDK de OpenCV en el Proyecto iOS

Para que Xcode compile el código C++ nativo de `segmentation.cpp` (que realiza el procesamiento digital de la imagen en 50ms), CocoaPods enlazará OpenCV:

1.  Descarga el SDK de **OpenCV para iOS** (un archivo comprimido `.zip`) desde el sitio oficial de OpenCV: [https://opencv.org/releases/](https://opencv.org/releases/) (se recomienda la versión 4.8.0 o superior).
2.  Descomprime el archivo descargado. Obtendrás una carpeta llamada `opencv2.framework`.
3.  Crea una carpeta en el proyecto iOS de Flutter: `derma-scan-app/ios/Frameworks/`.
4.  Copia la carpeta `opencv2.framework` y pégala dentro de la carpeta `Frameworks/` que acabas de crear.

---

## 📸 Paso 2: Preparar tu iPhone 13 para Desarrollo

1.  Conecta tu **iPhone 13** a tu Mac mediante el cable Lightning/USB-C.
2.  En el iPhone, desbloquea la pantalla y selecciona **"Confiar en esta computadora"** e ingresa tu código de seguridad.
3.  **Activar el Modo Desarrollador (Developer Mode) en tu iPhone:**
    *   Ve a **Configuración** -> **Privacidad y seguridad**.
    *   Desplázate hacia abajo hasta la sección **Modo desarrollador**.
    *   Activa la casilla **Modo desarrollador** y reinicia tu iPhone.
    *   Al encender, desbloquea y pulsa **"Activar"** en la ventana emergente.

---

## 💻 Paso 3: Configurar el Perfil de Firma en Xcode

Para poder instalar una aplicación en tu iPhone físico, Apple requiere que esté firmada con un perfil de aprovisionamiento:

1.  En tu terminal de Mac, descarga los paquetes de Flutter y genera la carpeta de Xcode:
    ```bash
    cd /Users/terrazasllanosfernando/Desktop/IA/py/derma-scan-app
    flutter pub get
    cd ios
    pod install
    ```
2.  Abre Xcode.
3.  Selecciona **File -> Open...** y abre el archivo `/Users/terrazasllanosfernando/Desktop/IA/py/derma-scan-app/ios/Runner.xcworkspace`.
4.  En la barra lateral izquierda de Xcode, haz clic sobre **Runner** (el icono de color azul en la parte superior).
5.  Ve a la pestaña **Signing & Capabilities** (Firma y Capacidades).
6.  Activa la casilla **Automatically manage signing** (Gestionar firma automáticamente).
7.  En el selector de **Team** (Equipo), haz clic e inicia sesión con tu **Apple ID** (tu correo y contraseña de iCloud).
8.  En **Bundle Identifier**, cambia el identificador a uno único para ti (por ejemplo: `com.fernando.dermascan.app`). Xcode generará de forma automática el certificado de firma gratuito.

---

## 🚀 Paso 4: Compilar e Instalar en tu iPhone 13

1.  En la parte superior de Xcode, al lado del botón de "Play", haz clic en el selector de dispositivos y **selecciona tu iPhone 13 físico**.
2.  Haz clic en el botón de **Play (Build and Run)** en Xcode, o ejecuta la compilación directamente desde tu terminal de Flutter:
    ```bash
    cd /Users/terrazasllanosfernando/Desktop/IA/py/derma-scan-app
    flutter run -d <ID_DE_TU_IPHONE>
    ```
3.  Xcode compilará los módulos nativos en C++, enlazará OpenCV, cargará el modelo fusionado unificado de 384 dimensiones de PCA en los assets y lo instalará en tu iPhone.

### ⚠️ Nota de seguridad de iOS en el primer inicio:
La primera vez que se abra la app, tu iPhone mostrará el mensaje *"Desarrollador empresarial no confiable"*. Para autorizarla:
*   Ve a **Configuración** -> **General** -> **Administración de dispositivos y VPN**.
*   Selecciona tu Apple ID bajo "Desarrollador de confianza".
*   Presiona **"Confiar en [Tu Correo]"** y confirma.

¡Listo! La aplicación **DermaScan AI** estará en tu pantalla de inicio del iPhone 13 y funcionará de manera **100% offline**, procesando imágenes de cámara y galería de forma local a máxima velocidad.
