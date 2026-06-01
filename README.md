# DAACC_IA
---
# Clasificación de Señales de Tráfico

Proyecto de **clasificación multiclase** de señales de tránsito utilizando Redes Neuronales Convolucionales (CNN).

---

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo capaz de identificar correctamente **7 tipos de señales de tráfico** comunes. Se utilizó un enfoque de aprendizaje profundo con TensorFlow/Keras, partiendo de un modelo CNN creado desde cero y aplicando técnicas de preprocesamiento y data augmentation.

**Precisión alcanzada en Test:** **92.58%**

---

## Tecnologías Utilizadas

- **Lenguaje:** Python 3.12
- **Framework:** TensorFlow / Keras
- **Modelo:** CNN convolucional desde cero (3 bloques Conv2D + MaxPooling)
- **Data Augmentation:** `ImageDataGenerator`
- **Entorno:** Kaggle Notebook (GPU Tesla P100)
- **Visualización:** Matplotlib

---

## Dataset

- **Fuente:** [Road Signs Dataset](https://www.kaggle.com/datasets/ziadghanem01/road-signs)
- **Número de clases:** 7
  - Keep Left
  - Keep Right
  - No Entry
  - Pedestrian Crossing
  - Stop Sign
  - Turn Left
  - Turn Right

- **Imágenes totales combinadas:** 2,969
- **Estrategia de balanceo:** Todas las clases reducidas a **119 imágenes** (para evitar sesgo del modelo)

- **División final:**
  - Train: 70%
  - Validation: 10%
  - Test: 20%

---

## Metodología

1. **Combinación de datos**: Se unieron las carpetas `train` y `val` en una sola carpeta (`all_signals`).
2. **Balanceo de clases**: Todas las clases fueron igualadas a 119 imágenes.
3. **Data Augmentation**: Rotación, desplazamiento, zoom y flip horizontal.
4. **Modelo**: CNN con 3 bloques convolucionales.
5. **Entrenamiento**: 30 épocas con `steps_per_epoch=100`.

---

## Resultados
**Mejor Accuracy obtenida:**
| Conjunto                  | Accuracy | Loss   |
|---------------------------|----------|--------|
| **Train** (Epoca 28)      |   93.40  | 20.05  |
| **Validation** (Epoca 24) |   95.60  | 16.04  |
| **Test**                  |   95.32  | 14.34  |


### Gráficas de Entrenamiento

![Accuracy](/DAACC_IA/Results/accuracy.png)
![Loss](/DAACC_IA/Results/loss.png)

---

## Estructura del Proyecto en kaggle y Google Colab

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

## Estructura del Proyecto

```bash
DAACC_IA/
├── Road_Signals.ipynb            # Notebook principal
├── results/
├── README.md
```

---

## Cómo Ejecutar el Proyecto

1. Clonar el repositorio:
   ```bash
   git clone 
   cd road-sign-classification
    ```
2. Ir al link de colab:
   ```bash
   https://drive.google.com/file/d/1WnmyhfcQ-aAz_27V_5awFO6Y8Tl_CP8R/view?usp=sharing
    ```

