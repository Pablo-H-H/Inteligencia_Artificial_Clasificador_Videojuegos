Clasificador de Videojuegos con Redes Neuronales Convolucionales  
**Del modelo secuencial** a una arquitectura del estado del arte con Transfer Learning

![InceptionResNetV2](https://img.shields.io/badge/Arquitectura-InceptionResNetV2-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-green) ![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.83%25-success)

Proyecto de aprendizaje profundo para clasificar capturas de pantalla de **6 videojuegos populares** (extensible a 21). Se compara un modelo CNN básico con una versión mejorada que incorpora **Transfer Learning**, **Fine‑Tuning** y una capa **LSTM**, siguiendo las metodologías de los artículos *Lubinus et al. (2021)* [1] y *COVID‑DSNet* [2].

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Videojuegos Clasificados](#videojuegos-clasificados)
3. [Generación y Selección del Set de Datos](#generación-y-selección-del-set-de-datos)
4. [Preprocesado de los Datos](#preprocesado-de-los-datos)
5. [Implementación del Modelo Base – Versión 1](#implementación-del-modelo-base--versión-1)
6. [Implementación del Modelo Mejorado – Versión 3 (Estado del Arte)](#implementación-del-modelo-mejorado--versión-3-estado-del-arte)
7. [Evaluación Inicial del Modelo (V1Base)](#evaluación-inicial-del-modelo-v1Base)
8. [Refinamiento del Modelo (Mejoras y Comparación)](#refinamiento-del-modelo-mejoras-y-comparación)
9. [Correcciones Documentadas](#correcciones-documentadas)
10. [Estructura del Proyecto](#estructura-del-proyecto)
11. [Instalación y Requisitos](#instalación-y-requisitos)
12. [Uso de los Modelos](#uso-de-los-modelos)
13. [Referencias](#referencias)

---

## Descripción del Proyecto

Este repositorio documenta la construcción, evaluación y evolución de un clasificador de imágenes de videojuegos utilizando **Redes Neuronales Convolucionales (CNN)**. Se desarrollaron dos versiones:

- **Modelo Base (V1):** CNN secuencial simple con una sola capa convolucional.
- **Modelo Mejorado (V2):** Arquitectura híbrida **InceptionResNetV2 + LSTM**, basada en *Transfer Learning* con pesos pre‑entrenados en ImageNet y una estrategia de entrenamiento en dos fases.

El objetivo principal es demostrar cómo la incorporación de técnicas del estado del arte – respaldadas por artículos científicos revisados por pares – mejora drásticamente la precisión y la robustez del modelo.

---

## Videojuegos Clasificados

El dataset original abarca **21 títulos** representativos de distintos géneros y estilos visuales. Para las pruebas se utilizó un subconjunto de **6 clases**, manteniendo la escalabilidad a las 21 originales.

| #  | Videojuego             |
|----|------------------------|
| 1  | Clash Royale           |
| 2  | Escape From Tarkov     |
| 3  | Minecraft              |
| 4  | Overwatch              |
| 5  | Rocket League          |
| 6  | Sea of Thieves         |

---

## Generación y Selección del Set de Datos

Las imágenes provienen del dataset público de Kaggle [Videogame Image Classification] [6], que contiene aproximadamente **6,500 screenshots por clase** (~37,000 en total). La riqueza visual de cada juego permite al modelo aprender patrones únicos como paletas de colores, interfaces y estilos artísticos.

### División del Dataset

Para una evaluación robusta se realizó una partición en tres conjuntos:

| Subconjunto  | Proporción | Imágenes (aprox.) | Finalidad                           |
|--------------|------------|--------------------|-------------------------------------|
| Entrenamiento| 80%        | ~30,000             | Ajuste de pesos del modelo          |
| Validación   | 10%        | ~3,500              | Ajuste de hiperparámetros y monitoreo |
| Prueba       | 10%        | ~3,500              | Evaluación final del rendimiento    |

La separación se realizó de manera estratificada para asegurar que cada partición contuviera ejemplos representativos de todas las clases. En la primera versión del modelo base también se disponía de estos tres conjuntos, aunque el énfasis de la evaluación recaía en validación y prueba.

---

## Preprocesado de los Datos

### Redimensionamiento

- **Modelo Base:** Imágenes redimensionadas a **300×300 píxeles**.
- **Modelo Mejorado:** **299×299 píxeles** – tamaño nativo requerido por `InceptionResNetV2` [3].

### Escalamiento / Normalización

| Modelo | Técnica | Rango final |
|--------|--------|-------------|
| Base   | `rescale=1./255` (división simple) | [0, 1] |
| Mejorado | `tf.keras.applications.inception_resnet_v2.preprocess_input` (estandarización con media/σ de ImageNet) | [-1, 1] |

La normalización específica de la red pre‑entrenada garantiza una mejor transferencia de conocimiento [2], [3].

### Aumento de Datos (Data Augmentation)

**Modelo Base:** Se aplicó aumento de datos para mejorar la robustez, usando la siguiente configuración:

```python
rotation_range=10,
width_shift_range=0.2,
zoom_range=0.3,
horizontal_flip=True
```
### Modelo Mejorado (Data Augmentation)

También se aplicó aumento de datos con parámetros ajustados:

- Rotación: hasta 15°
- Desplazamiento: 10%
- Zoom: 10%

---

## Implementación del Modelo Base – Versión 1

### Arquitectura

CNN secuencial simple implementada en TensorFlow/Keras.

| Capa | Parámetros |
|------|-----------|
| Conv2D(10, 3×3, activation='relu', input_shape=(300,300,3)) | 280 |
| Flatten() | 0 |
| Dense(256, activation='relu') | 227,338,496 |
| Dense(6, activation='softmax') | 1,542 |
| **Total** | **227,340,318** |

### Hiperparámetros

- Optimizador: **RMSprop**
- Learning rate: **2e-5**
- Función de pérdida: **categorical_crossentropy**
- Batch size: **128**
- Épocas: **10**

---

### Registro de Entrenamiento

| Época | Train Loss | Train Acc | Val Loss | Val Acc |
|------|-----------|-----------|----------|----------|
| 1 | 2.0217 | 0.4307 | 1.1849 | 0.6544 |
| 2 | 1.5905 | 0.5494 | 0.8639 | 0.7072 |
| 3 | 1.2926 | 0.6195 | 1.0480 | 0.6797 |
| 4 | 1.0926 | 0.6766 | 0.7464 | 0.7623 |
| 5 | 0.9104 | 0.7253 | 0.5806 | 0.8182 |
| 6 | 0.7871 | 0.7616 | 0.7097 | 0.7691 |
| 7 | 0.6812 | 0.7914 | 0.4459 | 0.8483 |
| 8 | 0.6104 | 0.8093 | 0.4065 | 0.8689 |
| 9 | 0.5543 | 0.8244 | 0.3307 | 0.8996 |
| 10 | 0.4986 | 0.8402 | 0.3042 | 0.9022 |

### Resultados Clave

- **Train Accuracy máx:** 84.02%
- **Validation Accuracy máx:** 90.22% 

El modelo mejora consistentemente sin sobreajuste severo, pero con capacidad limitada.

---

## Implementación del Modelo Mejorado – Versión 3 (Estado del Arte)

### Fundamento Teórico

- **Lubinus et al. (2021)**: extracción jerárquica + Global Average Pooling  
- **COVID-DSNet (2022)**: integración de LSTM para contexto  

---

### Arquitectura (Inception-Med-Classifier)

```text
Input (299, 299, 3)
↓
InceptionResNetV2 (congelada)
↓
GlobalAveragePooling2D
↓
Reshape → (1, feature_dim)
↓
LSTM (256)
↓
Dense (512) + BatchNorm + ReLU
↓
Dropout (0.3)
↓
Dense (6) + Softmax
```

## Componentes Clave

- **InceptionResNetV2**: 164 capas, ~56M parámetros pre-entrenados en ImageNet. Excelente extractor de características [3].  
- **Global Average Pooling**: Reduce cada mapa de características a un valor, previniendo sobreajuste [1].  
- **LSTM (256 unidades)**: Modelado contextual sobre el vector de características [2].  
- **Dropout (0.3) + BatchNormalization**: Regularización y estabilización del gradiente.  

---

## Hiperparámetros y Estrategia de Entrenamiento

- **Optimizador**: Adam  
- **Pérdida**: categorical_crossentropy  
- **Batch size**: 32  
- **Épocas**: 15 (Fase 1) + 10 (Fase 2)  

### Estrategia

- **Fase 1 – Transfer Learning**
  - Entrenamiento solo de la cabeza (LSTM + capas densas)
  - Learning rate: `1e-4`

- **Fase 2 – Fine-Tuning**
  - Descongelamiento de capas ≥ 600
  - Learning rate: `1e-6`

### Callbacks

- ModelCheckpoint  
- ReduceLROnPlateau (factor=0.5, paciencia=2)  
- EarlyStopping (paciencia=5–7)  

---

## Evaluación Inicial del Modelo V1Base

### Métricas

| Métrica | Definición |
|--------|----------|
| Accuracy | (TP + TN) / Total |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) |

### Resultados (Test)

- **Accuracy**: 90.83%
- **Precision**: 90.86% (macro)  
- **Recall**: 90.83% (macro)  

### Insight

El modelo no presenta sobreajuste significativo, pero su capacidad de extracción es limitada (<92%).

---

## Interpretación de Resultados

El modelo base captura patrones generales (colores, formas), pero falla en detalles finos (UI, texturas).  
El alto número de parámetros (~227M) no se traduce en rendimiento debido a ineficiencia en capas densas.

---

## Refinamiento del Modelo

### Diagnóstico V1

- **Train Acc**: 88.20%  
- **Val Acc**: 91.02%  
- **Parámetros**: 227M  
- **Entrenamiento**: alto costo (~570s/época)  
- **Regularización**: inexistente  

---

## Comparación V1_Base vs V3

| Componente | V1 (Base) | V3 (Mejorado) | Justificación |
|-----------|----------|--------------|--------------|
| Arquitectura | CNN simple | InceptionResNetV2 + LSTM | Mayor capacidad |
| Input | 300×300 | 299×299 | Compatibilidad |
| Normalización | rescale | preprocess_input | Mejor transferencia |
| Regularización | Ninguna | Dropout + BN | Menor overfitting |
| Learning Rate | 0.001 | 1e-4 / 1e-6 | Control fino |
| Parámetros | 227M | 56M | Eficiencia |
| Estrategia | 1 fase | 2 fases | Optimización progresiva |

---

## Comparación de Desempeño

| Modelo | Train Acc | Val Acc | Test Acc | Precision | Recall | Observaciones |
|--------|----------|--------|----------|----------|--------|--------------|
| V1 | 84.02% | 90.02% | 91.02% | 91.04% | 91.02% | Capacidad limitada |
| V3 | 99.91% | 99.83% | 99.83% | 99.86% | 99.83% | Arquitectura híbrida |

**Mejora:** +8.81 puntos porcentuales

---

## Correcciones Documentadas

- Migración a **Transfer Learning**
- Integración de **LSTM**
- Regularización (**Dropout + BatchNorm**)
- Normalización específica del modelo
- Entrenamiento en **dos fases**
- Implementación de callbacks
- Validación robusta (train/val/test)

---

## Estructura del Proyecto

```text id="q3mz9t"
videogame-classifier-cnn/
├── checkpoints_Base/
├── videogames-cnn-dataset_V3/
│   ├── train/
│   ├── validation/
│   └── test/
├── inception_med_classifier_final/
├── README.md
└── requirements.txt
```


## Instalación y Requisitos


```bash
pip install tensorflow==2.10 numpy opencv-python matplotlib scikit-learn seaborn pillow tkinterdnd2
```


- **GPU NVIDIA (opcional)**: CUDA 11.2 + cuDNN 8.1  
- **Dataset**: Disponible en Kaggle  

---

## Uso de los Modelos

### Evaluación del Modelo Mejorado (V2)

```bash
python ClasificadorUsoV3.py
```

Carga el modelo final y genera:

- Precisión  
- Pérdida  
- Matriz de confusión  
- Reporte sobre el conjunto de prueba  

---

### Interfaz Gráfica para Predicción Individual

```bash
python ClasificadorUsoInterfaz.py
```

Permite:

- Seleccionar una imagen de videojuego  
- Arrastrar y soltar en la interfaz  
- Obtener la clase predicha  
- Visualizar la confianza del modelo  
- Guardar en la carpeta correspondiente  

---

## Referencias

1. F. L. Badillo, C. A. R. Hernández, B. M. Narváez, and Y. E. A. Trillos, “Redes neuronales convolucionales: un modelo de Deep Learning en imágenes diagnósticas. Revisión de tema,” Revista Colombiana De Radiología, vol. 32, no. 3, pp. 5591–5599, Sep. 2021, doi: 10.53903/01212095.161.
2. H. C. Reis and V. Turk, “COVID-DSNet: A novel deep convolutional neural network for detection of coronavirus (SARS-CoV-2) cases from CT and Chest X-Ray images,” Artificial Intelligence in Medicine, vol. 134, p. 102427, Oct. 2022, doi: 10.1016/j.artmed.2022.102427.
3. “Videogame video classification,” Sep. 13, 2021. https://www.kaggle.com/datasets/juanmartinzabala/videogamesvideosdataset 

[Archivos grandes que no se pudieron subir nativamente a GitHub](https://drive.google.com/drive/folders/1FZBI_aNzpiJpNRAwOhjKbv6yyApCh05Q?usp=sharing)
