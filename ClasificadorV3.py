import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, optimizers
import numpy as np
import os

# Verificar versiones (deseadas: TF 2.10, Python 3.10.11)
print("TensorFlow version:", tf.__version__)
print("Python version:", __import__('sys').version)

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN Y PARÁMETROS
# ----------------------------------------------------------------------
IMG_SIZE = (299, 299)            # Tamaño requerido por InceptionResNetV2
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15               # Entrenamiento de la cabeza
EPOCHS_PHASE2 = 10               # Fine-tuning
NUM_CLASSES = 6                  # Normal, Bacteriana, Viral, COVID
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-6

# Rutas a los directorios de datos (estructura: train/class1, train/class2, ... val/...)
TRAIN_DIR = 'videogames-cnn-dataset_V3/train'
VAL_DIR   = 'videogames-cnn-dataset_V3/validation'

# ----------------------------------------------------------------------
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ----------------------------------------------------------------------
# Preprocesamiento específico de InceptionResNetV2 (normaliza a [-1, 1])
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=applications.inception_resnet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=applications.inception_resnet_v2.preprocess_input
)

train_generator = datagen_train.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',     # One-hot encoding para 4 clases
    shuffle=True
)

val_generator = datagen_val.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Muestra las clases detectadas
print("Clases:", train_generator.class_indices)

# ----------------------------------------------------------------------
# 3. DEFINICIÓN DEL MODELO "Inception-Med-Classifier"
# ----------------------------------------------------------------------
def build_inception_med_classifier(input_shape=(299, 299, 3), num_classes=4):
    """
    Construye el modelo con Transfer Learning usando InceptionResNetV2.
    La cabeza incluye GAP, LSTM, Dense(512), BatchNorm, Dropout y Softmax.
    """
    # --- Base pre-entrenada (congelada inicialmente) ---
    base_model = applications.InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Congelamos la base

    # --- Entrada de la imagen ---
    inputs = keras.Input(shape=input_shape)

    # Pasar por la base
    x = base_model(inputs, training=False)

    # --- Cabeza de clasificación ---
    # Global Average Pooling (reduce mapas de características a un vector)
    x = layers.GlobalAveragePooling2D()(x)
    # Guardar el número de características para el reshape (ej: 1536)
    feature_dim = x.shape[-1]

    # Reshape para LSTM: convertir vector en secuencia de "1 paso temporal" con feature_dim características.
    # Esto sigue la idea del paper COVID-DSNet, adaptada a la salida aplanada.
    # Forma: (batch, 1, feature_dim)
    x = layers.Reshape((1, feature_dim))(x)

    # LSTM para análisis secuencial de características (texturas)
    x = layers.LSTM(256, return_sequences=False)(x)

    # Capa densa con 512 neuronas, BatchNorm y ReLU
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Dropout para regularización (tasa 0.3 según recomendación)
    x = layers.Dropout(0.3)(x)

    # Capa de salida con softmax para 4 clases
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='Inception_Med_Classifier')
    return model

model = build_inception_med_classifier(num_classes=NUM_CLASSES)
model.summary()

# ----------------------------------------------------------------------
# 4. FASE 1: TRANSFER LEARNING (solo entrena la cabeza)
# ----------------------------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks útiles
callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(
        'best_phase1.h5', monitor='val_loss', save_best_only=True, mode='min'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
]

print("--- FASE 1: Entrenando la cabeza del modelo ---")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1
)

# ----------------------------------------------------------------------
# 5. FASE 2: FINE-TUNING (descongelar capas profundas y reentrenar)
# ----------------------------------------------------------------------
# Descongelar la base y volver a congelar solo las primeras capas (hasta la 600)
base_model = model.layers[1]  # La base es la segunda capa (índice 1)
base_model.trainable = True

# Congelar todas las capas hasta la número 600 (según recomendación)
for layer in base_model.layers[:600]:
    layer.trainable = False

# Verificar cuántas capas son entrenables
print(f"Capas entrenables después de descongelar: {sum([1 for l in base_model.layers if l.trainable])}")

# Recompilar con tasa de aprendizaje muy baja para fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        'best_phase2.h5', monitor='val_loss', save_best_only=True, mode='min'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
]

print("--- FASE 2: Fine-tuning de capas profundas ---")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2
)

# ----------------------------------------------------------------------
# 6. EVALUACIÓN FINAL Y GUARDADO DEL MODELO
# ----------------------------------------------------------------------
# Cargar el mejor modelo de la fase 2 (opcional si se usa restore_best_weights)
model.load_weights('best_phase2.h5')

# Evaluar en el conjunto de validación
results = model.evaluate(val_generator)
print(f"Pérdida final: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}, Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")

# Guardar el modelo final en formato SavedModel (recomendado)
model.save('inception_med_classifier_final')
print("Modelo guardado exitosamente.")