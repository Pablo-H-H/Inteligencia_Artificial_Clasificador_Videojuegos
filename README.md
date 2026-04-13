# Clasificador de Videojuegos con CNN

Modelo de Red Neuronal Convolucional (CNN) capaz de detectar y clasificar imágenes pertenecientes a **21 videojuegos distintos** a partir de screenshots y arte visual del juego.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Videojuegos Clasificados](#videojuegos-clasificados)
- [Generación y Selección del Set de Datos](#generación-y-selección-del-set-de-datos)
- [Preprocesado de los Datos](#preprocesado-de-los-datos)
- [Requisitos](#requisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)

---

## Descripción del Proyecto

Este proyecto implementa una Red Neuronal Convolucional (CNN) entrenada para clasificar imágenes visuales de videojuegos populares. Dado el estilo gráfico distintivo de cada título, el modelo aprende a identificar patrones visuales únicos —como paletas de colores, interfaces, entornos y estilos artísticos— para predecir a qué videojuego pertenece una imagen dada.

---

## Videojuegos Clasificados

El modelo es capaz de reconocer imágenes de los siguientes **21 videojuegos**:

| #  | Videojuego             | #  | Videojuego          |
|----|------------------------|----|---------------------|
| 1  | Apex Legends           | 12 | Minecraft           |
| 2  | CS:GO                  | 13 | Overwatch           |
| 3  | Clash Royale           | 14 | PUBG Battlegrounds  |
| 4  | Dead by Daylight       | 15 | Rainbow Six Siege   |
| 5  | Dota 2                 | 16 | Rocket League       |
| 6  | Escape from Tarkov     | 17 | Rust                |
| 7  | FIFA 21                | 18 | Sea of Thieves      |
| 8  | Fortnite               | 19 | Valorant            |
| 9  | Free Fire              | 20 | Warzone             |
| 10 | GTA V                  | 21 | World of Warcraft   |
| 11 | League of Legends      |    |                     |

---

## Generación y Selección del Set de Datos

### Obtención del Dataset

El dataset fue obtenido desde [Kaggle](https://www.kaggle.com/datasets/juanmartinzabala/videogame-image-classification), y está compuesto por **screenshots e imágenes de arte visual** de cada uno de los 21 videojuegos. Cada clase cuenta con aproximadamente **5,000 imágenes**, lo que resulta en un dataset total aproximado de **105,000 imágenes**.

La amplia cantidad de imágenes por clase asegura suficiente variabilidad visual para que el modelo generalice correctamente, por lo que **no se consideró necesario aplicar técnicas de aumento de datos (data augmentation)**.

> **Fuente:** Dataset de videojuegos disponible en Kaggle.

### Separación de los Sets de Entrenamiento y Prueba

El dataset completo fue dividido en dos subconjuntos utilizando una proporción **90/10**:

| Subconjunto  | Proporción | Uso                                   |
|--------------|------------|---------------------------------------|
| `train_data` | 90%        | Entrenamiento del modelo CNN          |
| `test_data`  | 10%        | Evaluación del rendimiento del modelo |

La división se realizó de forma **aleatoria y estratificada**, garantizando que cada clase quede representada proporcionalmente en ambos subconjuntos.

---

## Preprocesado de los Datos

### Escalamiento de Píxeles

Los valores de los píxeles de cada imagen fueron normalizados al rango **[0, 1]** dividiendo cada valor entre 255:

```python
imagen_normalizada = imagen / 255.0
```

Este escalamiento es fundamental para estabilizar el entrenamiento de la red neuronal, ya que evita que valores grandes dominen el cálculo del gradiente.

### Redimensionamiento de Imágenes

Todas las imágenes de entrada fueron redimensionadas a una resolución uniforme de **128 × 128 píxeles**, independientemente de su resolución original. Esto garantiza que todas las muestras tengan la misma dimensión de entrada requerida por la arquitectura CNN.

```python
imagen_redimensionada = cv2.resize(imagen, (128, 128))
```

### Resumen del Pipeline de Preprocesado

```
Imagen original
      │
      ▼
Redimensionamiento → 128x128 píxeles
      │
      ▼
Normalización → valores en rango [0, 1]
      │
      ▼
Tensor listo para la CNN
```

---

## Requisitos

```
Python >= 3.8
TensorFlow / Keras
NumPy
OpenCV (cv2)
Matplotlib
scikit-learn
```

Instalación de dependencias:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

---

## Estructura del Proyecto

```
videogame-classifier-cnn/
├── images/
│   ├── train/                        # Imágenes de entrenamiento (90%)
│   │   ├── Apex_Legends/
│   │   │   ├── img_0001.jpg
│   │   │   ├── img_0002.jpg
│   │   │   └── ...
│   │   ├── CSGO/
│   │   ├── Clash_Royale/
│   │   ├── Dead_by_Daylight/
│   │   ├── Dota2/
│   │   ├── Escape_From_Tarkov/
│   │   ├── FIFA21/
│   │   ├── Fortnite/
│   │   ├── FreeFire/
│   │   ├── GTAV/
│   │   ├── League_of_Legends/
│   │   ├── Minecraft/
│   │   ├── Overwatch/
│   │   ├── PUBG_Battlegrounds/
│   │   ├── Rainbow/
│   │   ├── Rocket_League/
│   │   ├── Rust/
│   │   ├── Sea_of_Thieves/
│   │   ├── Valorant/
│   │   ├── Warzone/
│   │   └── World_of_Warcraft/
│   └── test/                         # Imágenes de prueba (10%)
│       ├── Apex_Legends/
│       ├── CSGO/
│       ├── Clash_Royale/
│       ├── Dead_by_Daylight/
│       ├── Dota2/
│       ├── Escape_From_Tarkov/
│       ├── FIFA21/
│       ├── Fortnite/
│       ├── FreeFire/
│       ├── GTAV/
│       ├── League_of_Legends/
│       ├── Minecraft/
│       ├── Overwatch/
│       ├── PUBG_Battlegrounds/
│       ├── Rainbow/
│       ├── Rocket_League/
│       ├── Rust/
│       ├── Sea_of_Thieves/
│       ├── Valorant/
│       ├── Warzone/
│       └── World_of_Warcraft/
├── notebooks/
│   └── cnn_classifier.ipynb
├── models/
│   └── modelo_cnn.h5
├── README.md
└── requirements.txt
```

---

## Autor

Proyecto desarrollado como parte de un módulo de aprendizaje automático con redes neuronales convolucionales.
