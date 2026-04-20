import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import pandas as pd

def build_model(num_classes: int, input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    Transfer Learning con EfficientNetB0 preentrenado en ImageNet.
    - La base extrae features visuales ricas sin necesidad de millones de épocas.
    - Solo entrenamos el 'head' personalizado al inicio (fase 1).
    - Luego hacemos fine-tuning de las últimas capas de EfficientNet (fase 2).
    """
    # Base preentrenada, sin incluir el clasificador original de ImageNet
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Congelar la base en la fase 1
    base_model.trainable = False

    # Head personalizado para nuestras 21 clases
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # training=False mantiene BatchNorm congelado
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs), base_model


def get_callbacks(checkpoint_path: str) -> list:
    """
    Callbacks que solucionan los problemas del entrenamiento original:
    - ModelCheckpoint: guarda solo el mejor modelo
    - ReduceLROnPlateau: baja el LR cuando la validación se estanca (soluciona épocas 12-13)
    - EarlyStopping: detiene el entrenamiento si no hay mejora real (evita overfitting)
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,   # ← cambio clave
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Si val_accuracy no mejora en 3 épocas → divide el LR por 2
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Si val_accuracy no mejora en 7 épocas → detiene y restaura el mejor modelo
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, reduce_lr, early_stop]


def get_data_generators(train_dir: str, input_size=(224, 224)):
    """
    Data augmentation mejorado:
    - Se añade shear y channel_shift para mayor variedad visual
    - Mayor rotation_range (20° en vez de 15°)
    - preprocessing_function de EfficientNet para normalizar correctamente
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Normalización específica de EfficientNet
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range=20.0,   # Variaciones de color realistas
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=32,          # ← cambio: de 64 a 32
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=32,          # ← cambio: de 64 a 32
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator


def plot_history(history: tf.keras.callbacks.History):
    df = pd.DataFrame(history.history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df[['accuracy', 'val_accuracy']].plot(ax=axes[0])
    axes[0].set_title('Precisión del Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True)

    df[['loss', 'val_loss']].plot(ax=axes[1])
    axes[1].set_title('Pérdida del Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('curvas_entrenamiento.png', dpi=150)
    plt.show()
    print("Curvas guardadas en 'curvas_entrenamiento.png'")


def main():
    # ── 1. Configuración de GPU ──────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configurada y lista.")
        except RuntimeError as e:
            print(e)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # ── 2. Rutas ─────────────────────────────────────────────────────────────
    base_dir       = 'videogames-cnn-dataset'
    train_dir      = os.path.join(base_dir, 'train')
    checkpoint_dir = 'checkpoints_videojuegos'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'mejor_modelo.weights.h5')

    NUM_CLASSES = 21
    INPUT_SHAPE = (224, 224, 3)   # ← volvemos a 224, el 240 no aporta y gasta más VRAM
    INPUT_SIZE  = (224, 224)

    # ── 3. Generadores ───────────────────────────────────────────────────────
    train_generator, val_generator = get_data_generators(train_dir, INPUT_SIZE)

    # ── 4. Modelo (solo UNA vez) ──────────────────────────────────────────────
    model, base_model = build_model(NUM_CLASSES, INPUT_SHAPE)

    # ── 5. FASE 1: Entrenar solo el head ─────────────────────────────────────
    print("\n── FASE 1: Entrenando el head (base congelada) ──")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    model.summary()

    history_fase1 = model.fit(
        train_generator,
        epochs=15,                   # ← EarlyStopping parará antes si corresponde
        validation_data=val_generator,
        callbacks=get_callbacks(checkpoint_path),
        workers=1,
        use_multiprocessing=False
    )

    # Cargar los mejores pesos de la fase 1 antes de continuar
    model.load_weights(checkpoint_path)

    # ── 6. FASE 2: Fine-tuning ────────────────────────────────────────────────
    print("\n── FASE 2: Fine-tuning de capas superiores de EfficientNet ──")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )

    history_fase2 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=get_callbacks(checkpoint_path),
        workers=1,
        use_multiprocessing=False
    )

    # ── 7. Graficar ───────────────────────────────────────────────────────────
    combined = {}
    for key in history_fase1.history:
        combined[key] = history_fase1.history[key] + history_fase2.history[key]

    class CombinedHistory:
        def __init__(self, h): self.history = h

    plot_history(CombinedHistory(combined))
    print(f"\nProceso terminado. Mejor modelo guardado en: {checkpoint_path}")


if __name__ == '__main__':
    main()