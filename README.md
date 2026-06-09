# DAACC_IA

# Clasificación de Señales de Tráfico Mediante Redes Neuronales Convolucionales

Proyecto de **clasificación multiclase** de señales de tránsito utilizando Redes Neuronales Convolucionales (CNN).

---

# Resumen 

Este proyecto presenta un enfoque basado en Redes Neuronales Convolucionales (CNN) para la clasificación de señales viales en siete categorías: 
- Keep Left
- Keep Right
- No Entry
- Pedestrian Crossing
- Stop Sign
- Turn Left
- Turn Right

Así mismo se proporciona una interfaz web interactiva para la evaluación de imágenes en tiempo real con visualización de probabilidades, habilitada directamente dentro de notebooks de Google Colaboratory. Los resultados experimentales demuestran un rendimiento de clasificación efectivo sobre el conjunto de datos personalizado de señales viales.

Palabras clave: Clasificación de señales viales · Redes neuronales convolucionales · Aprendizaje por transferencia · Clasificación de imágenes · Vehículos autónomos · Google Colab · Kaggle

---

# Estructura del Proyecto en kaggle y Google Colab

```bash
Road_Signals/
├── kaggle/
    ├── input/
        ├── datasets/
            ├── ziadghanem01/
                ├── road-signs/
                    ├── dataset/  # Dataset original
├── all_signals/                  # Dataset combinado
├── signals/                      # Dataset 
```

# Estructura del Proyecto

```bash
DAACC_IA/
├── Road_Signals.ipynb            # Notebook principal — entrenamiento y evaluación
├── Interfaz_road_signals.ipynb   # Notebook para prueba de interfaz interactiva
├── results/                      # Imágenes de resultados y métricas
│   ├── .png
│   ├── .png
│   └── .png
├── models/                       # Modelos guardados
│   └── road_signals_model_hybrid.h5
│   └── road_signals_model_paper.h5
│   └── road_signals_model.h5
└── README.md
```

---

# Tecnologías Utilizadas

- **Lenguaje:** Python 3.12
- **Framework:** TensorFlow / Keras
- **Modelo:** CNN convolucional desde cero (3 bloques Conv2D + MaxPooling)
- **Data Augmentation:** `ImageDataGenerator`
- **Entorno:** Kaggle Notebook (GPU Tesla P100)
- **Visualización:** Matplotlib

---

# Dataset

El conjunto de datos utilizado en este proyecto consiste en imágenes pertenecientes a siete clases de señales viales:

| ID de Clase | Etiqueta | Descripción |
|:---:|---|---|
| 0 | Keep Left | Señal de mantener la izquierda |
| 1 | Keep Right | Señal de mantener la derecha |
| 2 | No Entry | Prohibición de entrada |
| 3 | Pedestrian Crossing | Paso de peatones |
| 4 | Stop Sign | Señal de alto |
| 5 | Turn Left | Giro a la izquierda |
| 6 | Turn Right | Giro a la derecha |

**Imágenes totales:** 2,969

**Fuente:** [Road Signs Dataset](https://www.kaggle.com/datasets/ziadghanem01/road-signs)

---

# Preprocesamiento

- **Redimensionamiento:** Todas las imágenes redimensionadas a 224 × 224 píxeles para coincidir con la forma de entrada del modelo.

- **Normalización:** Valores de píxeles escalados al rango [0, 1] dividiendo entre 255.

- **Aumento de datos (Data Augmentation):** Aplicado únicamente al conjunto de entrenamiento para incrementar la variabilidad de los datos y reducir el sobreajuste:
  - Desplazamiento horizontal aleatorio de hasta un 20% (`width_shift_range=0.2`).
  - Desplazamiento vertical aleatorio de hasta un 20% (`height_shift_range=0.2`).
  - Transformación de cizallamiento (*shear*) de hasta un 20% (`shear_range=0.2`).
  - Zoom aleatorio de hasta un 20% (`zoom_range=0.2`).
  - No se aplicó volteo horizontal (`horizontal_flip=False`), ya que podría alterar el significado de algunas señales de tránsito.

- **Estrategia de balanceo:** Todas las clases reducidas a **119 imágenes** (para evitar sesgo del modelo)

- **División final:**
  - Train: 70%
  - Validation: 10%
  - Test: 20%
Las imágenes fueron preprocesadas mediante el siguiente proceso:

---

# Metodología

1. **Combinación de datos**: Se unieron las carpetas `train` y `val` en una sola carpeta (`all_signals`).
2. **Balanceo de clases**: Todas las clases fueron igualadas a 119 imágenes, con el fin de evitar el "sesgo del modelo", ya que el modelo puede aprende mucho mejor la clase que tiene más imagenes.
3. **Data Augmentation**: Rotación, desplazamiento, deformación y zoom.

## CNN Base - Arquitectura desde 0

La arquitectura CNN propuesta sigue un diseño secuencial con tres bloques convolucionales, cada uno compuesto por una capa de convolución seguida de max pooling, seguidos de capas completamente conectadas para la clasificación.
 
```
Entrada (224 × 224 × 3)
    │
    ├─ Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
    ├─ Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
    ├─ Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
    │
    ├─ Flatten
    ├─ Dense(512, ReLU)
    └─ Dense(7, Softmax)
 
Salida: Distribución de probabilidad sobre 7 clases
```

![alt text](results/base_model/descripcion.png)

**Configuración**:
    - **Optimizador:** Adam.
    - **Función de Pérdida:** Entropía cruzada categórica (`categorical_crossentropy`), adecuada para la clasificación multiclase con etiquetas en codificación *one-hot*.
    - **Métrica Principal:** Precisión (*Accuracy*).
    - **Flujo de Entrenamiento:** El proceso se extendió por un máximo de 30 épocas utilizando generadores de datos en lotes (*data streams*). Con el fin de regularizar y estructurar los ciclos de cómputo, se fijaron 100 pasos por época (`steps_per_epoch=100`) para el subconjunto de entrenamiento y 50 pasos para el subconjunto de validación (`validation_steps=50`).
3. **Enfoque Comparativo**: Aprendizaje por Transferencia (modelo del documenot `Road Sign Classification Using Transfer Learning and Pre-trained CNN Models`)



---

## Resultados
**Mejor Accuracy obtenida:**
| Conjunto                  | Accuracy | Loss   |
|---------------------------|----------|--------|
| **Train** (Epoca 28)      |   93.40  | 20.05  |
| **Validation** (Epoca 24) |   95.60  | 16.04  |
| **Test**                  |   95.32  | 14.34  |


### Gráficas de Entrenamiento

![Accuracy](Results/accuracy.png)
![Loss](Results/loss.png)

---

## Cómo Ejecutar el Proyecto

1. Clonar el repositorio:
   ```bash
   git clone 
   cd DAACC_IA
    ```
2. Ir al link de colab:
   ```bash
   https://drive.google.com/file/d/1WnmyhfcQ-aAz_27V_5awFO6Y8Tl_CP8R/view?usp=sharing
    ```

## Comparación con modelo
En el documento `Road Sign Classification Using Transfer Learning and Pre-trained CNN Models` no se construyo una red neuronal desde cero. En su lugar, utiliza una técnica llamada Transfer Learning (Aprendizaje por Transferencia). El cual consiste en tomar arquitecturas de modelos muy profundos que ya fueron pre-entrenados con millones de imágenes y adaptarlos a un nuevo problema (en este caso, las señales de la carretera). Los autores evaluaron 4 arquitecturas: VGG-16, VGG-19, ResNet50 y EfficientNetB0. Según sus resultados (Tabla 1 del documento `Performance Metrics of Road Sign Classification Models`), el modelo que mejor desempeño tuvo fue VGG-16, alcanzando un 99.21% de accuracy y un 99.11 de F1-score. Las configuraciones exactas que usaron los autores son: 
- Cargar el modelo pre-entrenado y congelar sus capas, asu vez, estas las utilizaron como extractores de características para la clasificación de señales de tráfico.
- Al final añadieron una ultima capa (una capa densa) conectada al modelo preentrenado, que toma las características extraídas como entrada y genera la distribución de probabilidad sobre las 43 clases de señales de tráfico.
- Hiperparámetros:
    - Optimizador: Adam
    - Tasa de aprendizaje (Learning Rate): $1e^{-5}$
    - Tamaño de lote (Batch size): 32 
    - Épocas: 15
    - Detención temprana (Early Stopping): Paciencia de 3 épocas si la validación no mejora.

Y a su vez ellos usaron el dataset: GTSRB (alemán, 43 clases)

## Referencias
[1]
‌Hosseini, S.H., Ghaderi, F., Moshiri, B., Norouzi, M. (2023). Road Sign Classification Using Transfer Learning and Pre-trained CNN Models.