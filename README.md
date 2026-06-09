# DermaScan AI — Clasificador CNN de Lesiones de Piel (v2.0)

Este repositorio contiene el pipeline de Machine Learning (ML) de nivel de producción de **DermaScan AI**, un sistema basado en **Redes Neuronales Convolucionales (CNN)** diseñado para la clasificación automatizada y offline de lesiones dermatológicas en tres clases críticas.

---

## 1. El Problema que Soluciona el Proyecto

El cáncer de piel es uno de los cánceres más comunes a nivel mundial. Su detección temprana es fundamental para la supervivencia del paciente, especialmente en casos de **Melanoma**, el cual tiene una tasa de mortalidad elevada si no se diagnostica a tiempo.

DermaScan AI soluciona este problema proporcionando una herramienta de cribado (screening) rápido y de bajo costo que funciona de manera **totalmente local y offline en dispositivos móviles**, permitiendo un pre-diagnóstico confiable de las siguientes lesiones:

| Clase | Nombre Completo | Tipo de Lesión | Significado Clínico / Gravedad |
|---|---|---|---|
| **AK** | Queratosis Actínica | Pre-maligna | Parches escamosos causados por el sol; pueden derivar en carcinoma si no se tratan. |
| **BCC** | Carcinoma Basocelular | Maligna (Cáncer) | El cáncer de piel más común; de crecimiento lento, localmente invasivo pero rara vez metastásico. |
| **MEL** | Melanoma | Maligna (Cáncer) | El tipo de cáncer de piel más agresivo y letal; alta capacidad de metástasis si no se detecta temprano. |

---

## 2. Stack Tecnológico

El proyecto está construido sobre tecnologías de vanguardia en Inteligencia Artificial y computación móvil para garantizar la reproducibilidad y el despliegue óptimo:

* **Framework de Deep Learning:** PyTorch (v2) y Torchvision (entrenamiento del modelo y carga de datos).
* **Hardware Acceleration:** Apple MPS (Metal Performance Shaders) para aceleración en chips Apple Silicon (M1/M2/M3), compatible también con NVIDIA CUDA y CPU.
* **Procesamiento de Imágenes:** Pillow (PIL) y OpenCV (manipulación e inferencia).
* **Evaluación y Métricas:** Scikit-Learn (cálculo de F1-Score, matrices de confusión, curvas ROC y curvas de Precisión-Recall).
* **Visualización:** Matplotlib (generación automática de reportes gráficos y curvas de aprendizaje).
* **Formato de Exportación:** ONNX (Open Neural Network Exchange) para empaquetar el grafo y desplegarlo en el dispositivo móvil.
* **Plataforma de Despliegue:** Flutter (aplicación móvil que ejecuta el modelo ONNX en tiempo real sin requerir internet).

---

## 3. Arquitectura de Inferencia Móvil: Flujo de Dos Modelos (Dual-Model Flow)

Para lograr un pre-diagnóstico confiable y autónomo directamente en el dispositivo móvil, la aplicación DermaScan AI ejecuta un **flujo de inferencia secuencial compuesto por dos modelos de Deep Learning independientes**:

```
[Foto de la Piel con Cámara] 
        ↓
1. MODELO SEGMENTADOR (U-Net) ➔ Recorta la lesión y elimina vello/ruido de la piel sana
        ↓ (Retorna imagen recortada de 260x260 con fondo negro)
2. MODELO CLASIFICADOR (EfficientNet-B2) ➔ Predice probabilidades para AK, BCC o MEL
```

### A. Modelo 1: Segmentador U-Net (`fused_segmenter_unet.onnx`)
* **Propósito:** Localizar la lesión sospechosa dentro de la fotografía tomada por el usuario, recortarla de forma ajustada e ignorar el entorno irrelevante (por ejemplo, vellos, reflejos de luz y piel sana circundante).
* **Características:** Es un modelo sumamente ligero (**~525 KB** en formato ONNX) adaptado de la clásica arquitectura U-Net. Su función es actuar como filtro de entrada, asegurando que el modelo clasificador solo reciba la región de interés (lesión aislada sobre fondo negro), lo cual previene significativamente los falsos positivos por confusión con características de la piel sana.

### B. Modelo 2: Clasificador EfficientNet-B2 (`fused_dermatology_model.onnx`)
* **Propósito:** Analizar la lesión pre-segmentada proveniente del Modelo 1 y predecir la probabilidad de que corresponda a Queratosis Actínica (AK), Carcinoma Basocelular (BCC) o Melanoma (MEL).
* **Características:** Es la red neuronal profunda entrenada en este repositorio (**34.7 MB**). Utiliza la capacidad de extracción de características visuales de EfficientNet-B2 combinada con el cabezal multicapa regularizado para proporcionar el pre-diagnóstico final con alta precisión métrica.

> **Nota Académica para la Sustentación:** Esta división de tareas (separar localización/segmentación de clasificación) es una de las mejores prácticas en el desarrollo de IA médica. Al aislar la lesión, garantizamos que el clasificador se enfoque exclusivamente en los patrones patológicos de la lesión (red pigmentada, glóbulos, estructuras vasculares) y no en el tono de piel del paciente, aumentando la equidad y robustez del sistema en diferentes fototipos cutáneos.

---

## 4. Pipeline de Machine Learning (ML Pipeline) Paso a Paso

El ciclo de vida de desarrollo del modelo sigue una arquitectura estructurada y rigurosa:

```
[Imágenes Segmentadas] ➔ [Preprocesamiento & Balanceo] ➔ [Feature Extraction (B2)] ➔ [Fine-Tuning (AdamW)] ➔ [Evaluación Ciega] ➔ [Exportación ONNX] ➔ [Flutter Assets]
```

### Paso 1: Preprocesamiento y Aumentación de Datos
* **Segmentación Clínica:** Las imágenes provienen de un preprocesamiento clásico en C++ (remoción de vello, realce CLAHE, máscara Otsu multicanal y operaciones morfológicas) que recorta la lesión sobre un fondo negro para eliminar ruido de la piel sana.
* **Redimensionamiento:** Todas las imágenes se reescalan a **260×260 píxeles** (el tamaño óptimo de entrada para EfficientNet-B2).
* **Balanceo de Clases:** El dataset original contiene una distribución desbalanceada. Se aplica un **submuestreo aleatorio (Undersampling)** controlado por semilla reproducible en el entrenamiento para igualar la distribución a **1,614 imágenes por clase** (4,842 muestras en total), previniendo sesgos hacia clases mayoritarias.
* **Aumentación de Datos Clínica:** A diferencia de proyectos genéricos, en imágenes dermatoscópicas tomadas con lentes de contacto fijos, las alteraciones fotométricas (iluminación, desenfoque) son innecesarias y distorsionan la lesión real. Por ende, solo se aplican transformaciones **geométricas puras**:
  - Giros horizontales y verticales aleatorios (`RandomHorizontalFlip`, `RandomVerticalFlip`, `p=0.5`).
  - Rotaciones leves (`RandomRotation`, máximo 15°).
  Esto entrena al modelo para ser invariante a la orientación física en la que se capture la fotografía.
* **Normalización:** Se estandarizan los canales R, G y B con las estadísticas de ImageNet (Media: `[0.485, 0.456, 0.406]`, Desviación Estándar: `[0.229, 0.224, 0.225]`).

### Paso 2: Extracción de Características (Feature Extraction)
* **Model Selection:** Se seleccionó **EfficientNet-B2** sobre arquitecturas tradicionales como ResNet debido a su escalado compuesto optimizado (ancho, profundidad y resolución del input escalados uniformemente).
* **Feature Extractor:** La red aprovecha el conocimiento previo del modelo preentrenado en millones de imágenes de ImageNet. Durante la primera etapa del entrenamiento, el backbone convolucional (capas de extracción) está congelado, capturando texturas básicas, bordes y patrones visuales generales sin modificar sus pesos preentrenados (7.7 millones de parámetros congelados).

### Paso 3: Selección del Modelo y Entrenamiento (Model Selection & Training)
* **Cabezal de Clasificación Personalizado (Custom Head):** En lugar de proyectar directamente las 1408 características del backbone a las 3 clases (lo que provocaría pérdida severa de información), se diseñó un cabezal clasificador intermedio:
  - `Dropout` primario de **0.35** para regularizar.
  - Capa densa intermedia `Linear(1408 ➔ 512)` para proyectar las características a un espacio dermatológico específico.
  - Normalización por lotes (`BatchNorm1d`) para estabilizar las activaciones y acelerar la convergencia.
  - Activación `SiLU` (Swish), consistente con la arquitectura interna de EfficientNet.
  - `Dropout` secundario de **0.35** para evitar la co-adaptación de neuronas.
  - Capa de salida `Linear(512 ➔ 3)` que proyecta los logits para AK, BCC y MEL.
* **Entrenamiento en Dos Etapas (Two-Stage Fine-Tuning):**
  - **Fase 1 (Épocas 1-5):** Solo se entrena el cabezal personalizado con un Learning Rate relativamente alto (`LR = 0.001`) y el backbone convolucional congelado. Esto estabiliza la cabeza clasificadora antes de perturbar los pesos del extractor.
  - **Fase 2 (Época 6 en adelante):** Se descongela toda la red y se realiza un ajuste fino (Full Fine-Tuning) con un Learning Rate extremadamente conservador (`LR = 5e-5`) para adaptar las capas profundas a la estructura fina de las lesiones cutáneas sin destruir el conocimiento previo.
* **Optimizador:** **AdamW** (con decaimiento de peso / Weight Decay = `1e-2`), el cual desacopla la regularización L2 del gradiente, penalizando pesos excesivamente grandes para controlar el sobreajuste.
* **Función de Pérdida:** Entropía cruzada con **Label Smoothing (0.1)**. Esta técnica redistribuye un 10% de la confianza a otras clases durante el cálculo de la pérdida, evitando que la red se vuelva demasiado confiada en sus predicciones y mejorando la calibración y generalización en bordes de decisión difusos.
* **Learning Rate Scheduler:** Se utiliza `ReduceLROnPlateau` que reduce el Learning Rate a la mitad si la pérdida de validación no mejora tras un número determinado de épocas (`patience=4` en Stage 2), permitiendo un descenso suave y preciso al final del entrenamiento.

### Paso 4: Validación del Modelo (Model Validation)
* **División Estratificada:** El conjunto de datos se divide de forma balanceada y sin fuga de información (no data leakage):
  - **70%** para Entrenamiento.
  - **15%** para Validación (usado para control y políticas de parada).
  - **15%** para Prueba (Test ciego, utilizado únicamente en la evaluación final).
* **Early Stopping (Parada Temprana):** Se monitorea activamente la pérdida de validación (`Val Loss`) en la Fase 2. Si la pérdida de validación no mejora durante **7 épocas consecutivas**, el entrenamiento se detiene automáticamente y se recuperan los pesos del modelo que obtuvieron el mejor rendimiento histórico en validación. Esto impide de forma estricta que la red memorice ruido del conjunto de entrenamiento.

### Paso 5: Evaluación Científica (Model Evaluation)
Una vez finalizado el entrenamiento, el script de análisis genera reportes científicos de evaluación:
* **Matriz de Confusión:** Revela la cantidad exacta de verdaderos positivos, falsos positivos y falsos negativos cruzados por clase.
* **Reporte de Clasificación:** Computa métricas clave como:
  - **Precisión (Precision):** ¿Qué porcentaje de las lesiones que el modelo clasificó como X eran verdaderamente X? (Crítico para minimizar falsos positivos en Melanoma).
  - **Recall (Sensibilidad):** ¿Qué porcentaje de las lesiones reales X fue capaz de capturar el modelo? (Crítico para que ningún cáncer pase desapercibido).
  - **F1-Score:** La media armónica entre Precisión y Sensibilidad, indicando la robustez global del modelo.
* **Curvas ROC-AUC:** Mide el rendimiento a diferentes umbrales de decisión. El Área bajo la Curva (AUC) cercana a 1.0 valida la capacidad sobresaliente de discriminación del modelo.
* **Curvas Precision-Recall:** Útil para evaluar el rendimiento bajo el foco del balance clínico.
* **Gráficos de Caja de Confianza (Confidence Boxplots):** Demuestra que el modelo tiene alta confianza (ej. >90%) en predicciones correctas y baja confianza en predicciones incorrectas, indicando buena calibración.
* **Curvas de Aprendizaje (Bias vs Variance):** Valida que el tamaño del dataset es adecuado y que no hay una brecha excesiva entre entrenamiento y validación (varianza controlada).

### Paso 6: Despliegue en Dispositivo Móvil (Deployment)
* **Fused Inference Graph (FusedDermatologyCNN):** Para evitar que la aplicación móvil tenga que implementar transformaciones complejas de imágenes que difieran del preprocesamiento de PyTorch, creamos una envoltura de inferencia unificada en PyTorch:
  1. Recibe un vector plano de píxeles normalizados entre 0.0 y 1.0 (forma: `[1, 202800]` para 260x260x3).
  2. Redimensiona internamente el vector plano a un tensor 3D de imagen (`[1, 3, 260, 260]`).
  3. Aplica la normalización matemática de ImageNet usando buffers internos.
  4. Ejecuta el paso forward por la red EfficientNet-B2 cargada en memoria.
  5. Aplica la función `Softmax` final para retornar un vector de probabilidades de tamaño 3 (`[AK, BCC, MEL]`).
* **Compilación y ONNX:** Todo este flujo se compila en un archivo único de formato ONNX (`fused_dermatology_model.onnx`, peso: **34.7 MB**), autocontenido, libre de dependencias de PyTorch y listo para ser interpretado offline en iOS y Android a través del SDK de Flutter utilizando `onnxruntime` o `tflite_flutter`.

---

## 5. Métricas de Rendimiento del Modelo Definitivo

Las siguientes son las métricas reales y auditadas del modelo definitivo de producción evaluado sobre el **Test set ciego (727 imágenes independientes)**:

### Resultados Globales:
* **Exactitud Global (Accuracy):** **92.30%**
* **Precisión Promedio (Macro Average Precision):** **92.36%**
* **Recall Promedio (Macro Average Recall):** **92.30%**
* **F1-Score Promedio (Macro Average F1):** **92.31%**
* **Comportamiento en Entrenamiento:** Convergencia limpia y libre de sobreajuste (Train Accuracy: **99.2%**, Val Accuracy estable: **~92%**).

### Reporte de Clasificación por Clase (Test Set):
```
              precision    recall  f1-score   support

           AK     0.9073    0.9298    0.9184       242
          BCC     0.9065    0.9215    0.9139       242
          MEL     0.9571    0.9177    0.9370       243

     accuracy                         0.9230       727
    macro avg     0.9236    0.9230    0.9231       727
```

### Conclusiones Clave para Sustentar el Proyecto:
1. **Precisión Sobresaliente en Melanoma (95.71%):** Esto significa que cuando el modelo diagnostica un Melanoma, la probabilidad de que sea un falso positivo es de apenas **4.29%**. Clínicamente, esto evita la angustia innecesaria del paciente y optimiza los recursos de derivación médica.
2. **Alta Sensibilidad (Recall) General:** Una sensibilidad de **~92%** en todas las clases asegura que el sistema captura la gran mayoría de las lesiones malignas, minimizando los falsos negativos (casos donde hay cáncer pero el modelo no lo detecta).
3. **Robustez y Eficiencia del Tamaño:** Con solo **34.7 MB** de tamaño de archivo ONNX, el modelo corre de manera fluida y fluida en dispositivos móviles comerciales sin requerir almacenamiento masivo ni hardware de servidor.
4. **Generalización Excepcional:** La ínfima diferencia entre la exactitud de validación (~92%) y prueba (92.30%) demuestra la efectividad de la parada temprana, el decaimiento de peso y el dropout doble aplicados en el diseño.

---

## 6. Instrucciones de Uso

### Instalación de Dependencias
Asegúrate de contar con Python 3.10 o superior y ejecuta:
```bash
pip install -r requirements.txt
```

### Ejecutar el Entrenamiento
Para entrenar el modelo desde cero con el pipeline de dos etapas y exportarlo automáticamente:
```bash
python train.py
```
*Este comando generará el modelo `outputs/best_model.pth` y exportará el archivo unificado `outputs/fused_dermatology_model.onnx`.*

### Validar el Modelo en el Conjunto de Prueba
Para validar el modelo final sobre el conjunto ciego de prueba de forma independiente:
```bash
python validate_model_on_oficial.py --test-only
```

### Generar Reportes Científicos
Para recrear los gráficos de análisis (ROC, Precision-Recall, diagramas de caja de confianza y curvas de aprendizaje):
```bash
python generate_analysis_plots.py
```
