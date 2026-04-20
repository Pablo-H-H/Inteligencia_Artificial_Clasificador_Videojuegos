import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Hiperparámetros globales ───────────────────────────────────────
NUM_CLASSES  = 3
INPUT_SHAPE  = (250, 250, 3)  # EfficientNetB3 nativo en 300x300
INPUT_SIZE   = (250, 250)
BATCH_SIZE   = 128             # Colab T4 tiene 16GB VRAM, podemos usar 32
EPOCHS_F1    = 20             # Fase 1: solo el head
EPOCHS_F2    = 25             # Fase 2: fine-tuning
MIXUP_ALPHA  = 0.2

# ── Configura tus rutas aquí ───────────────────────────────────────
DRIVE_BASE     = ''
DATASET_DIR    = os.path.join(DRIVE_BASE, 'videogames-cnn-dataset_V2')
TRAIN_DIR      = os.path.join(DATASET_DIR, 'train')
CHECKPOINT_DIR = os.path.join(DRIVE_BASE, 'checkpoints_videojuegos_V2')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'mejor_modelo.weights.h5')
# ──────────────────────────────────────────────────────────────────

def get_data_generators(train_dir, input_size=INPUT_SIZE, batch_size=BATCH_SIZE):
    """
    Crea generadores de entrenamiento y validación.
    - preprocessing_function: normalización específica de EfficientNet
    - Augmentation: rotación, desplazamiento, zoom, flip, channel_shift
    - validation_split: 20% para validación
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range=20.0,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen

def mixup_generator(generator, alpha=MIXUP_ALPHA, batch_size=BATCH_SIZE):
    """
    Generador infinito con MixUp que descarta automáticamente
    lotes incompletos para evitar errores de broadcasting.
    """
    while True:
        # Obtener primer lote
        X1, y1 = next(generator)
        # Si el lote no está completo, lo ignoramos y pedimos otro
        if X1.shape[0] != batch_size:
            continue
        
        # Obtener segundo lote
        X2, y2 = next(generator)
        # Asegurar que también esté completo
        while X2.shape[0] != batch_size:
            X2, y2 = next(generator)
        
        # Mezclar
        lam = np.random.beta(alpha, alpha)
        X_mix = lam * X1 + (1 - lam) * X2
        y_mix = lam * y1 + (1 - lam) * y2
        yield X_mix, y_mix

def build_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE):
    """
    EfficientNetB3 + head personalizado.
    Mejoras vs versión anterior (B0):
      - B3: 12M params vs 5.3M de B0 → más capacidad representacional
      - L2 regularization: penaliza pesos grandes en Dense
      - Dropout 0.5/0.4 (antes 0.4/0.3): más regularización
    """
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Congelado en Fase 1

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        256, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.4)(x)

    # softmax en float32 explícito (recomendado con mixed_float16)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    return tf.keras.Model(inputs, outputs), base_model

def get_callbacks(checkpoint_path):
    """
    ModelCheckpoint : guarda solo pesos cuando val_accuracy mejora
    ReduceLROnPlateau: divide LR por 2 si no mejora en 3 épocas
    EarlyStopping   : para si no mejora en 7 épocas
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    return [checkpoint, reduce_lr, early_stop]



def main():
    # ── Configuración de GPU ───────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurada con memory growth")
        except RuntimeError as e:
            print(e)
    

    # Precisión mixta: float16 en GPU → ~2x más rápido en T4/A100
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Política de precisión: {tf.keras.mixed_precision.global_policy().name}")



    print(f"\nConfiguración:")
    print(f"  Input shape : {INPUT_SHAPE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Clases      : {NUM_CLASSES}")
    print(f"  Épocas F1   : {EPOCHS_F1}")
    print(f"  Épocas F2   : {EPOCHS_F2}")

    train_generator, val_generator = get_data_generators(TRAIN_DIR)

    print(f"\nImagenes de entrenamiento : {train_generator.samples}")
    print(f"Imágenes de validación    : {val_generator.samples}")
    print(f"Clases encontradas        : {list(train_generator.class_indices.keys())}")

    # Mostrar ejemplos de imágenes del dataset
    class_names = list(train_generator.class_indices.keys())
    X_sample, y_sample = next(train_generator)

    # EfficientNet preprocess_input modifica el rango de píxeles,
    # así que normalizamos para visualización
    def denormalize(img):
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        return img

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i, ax in enumerate(axes.flat):
        if i < len(X_sample):
            ax.imshow(denormalize(X_sample[i]))
            ax.set_title(class_names[np.argmax(y_sample[i])], fontsize=8)
        ax.axis('off')
    plt.suptitle('Ejemplos del Dataset (con augmentation)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # Visualizar efecto de MixUp
    X1, y1 = next(train_generator)
    X2, y2 = next(train_generator)
    lam = 0.5  # mezcla 50/50 para visualizar claramente
    X_mix = lam * X1 + (1 - lam) * X2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(denormalize(X1[0]));  axes[0].set_title(f'Imagen A: {class_names[np.argmax(y1[0])]}');  axes[0].axis('off')
    axes[1].imshow(denormalize(X2[0]));  axes[1].set_title(f'Imagen B: {class_names[np.argmax(y2[0])]}');  axes[1].axis('off')
    axes[2].imshow(denormalize(X_mix[0])); axes[2].set_title(f'MixUp (λ=0.5)'); axes[2].axis('off')
    plt.suptitle('Efecto de MixUp Augmentation', fontsize=13)
    plt.tight_layout()
    plt.show()
    print("En entrenamiento real, λ es aleatorio (distribución Beta), no fijo en 0.5")

    model, base_model = build_model()
    model.summary()

    total_params    = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nParámetros totales     : {total_params:,}")
    print(f"Parámetros entrenables : {trainable_params:,} (solo el head en Fase 1)")

    print("✅ Callbacks definidos:")
    print(f"  Checkpoint → {CHECKPOINT_PATH}")
    print( "  ReduceLROnPlateau → patience=3, factor=0.5")
    print( "  EarlyStopping     → patience=7")

    print("── FASE 1: Entrenando el head (base congelada) ──\n")

    model.compile(
        # Label Smoothing 0.1: penaliza predicciones demasiado confiadas
        # Reportado efectivo en la investigación con DenseNet en UC Merced
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # Generador con MixUp para entrenamiento
    train_mixed_f1 = mixup_generator(train_generator, alpha=MIXUP_ALPHA, batch_size=BATCH_SIZE)
    steps_per_epoch = train_generator.samples // BATCH_SIZE

    history_fase1 = model.fit(
        train_mixed_f1,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_F1,
        validation_data=val_generator,
        callbacks=get_callbacks(CHECKPOINT_PATH),
        workers=1,              # Colab permite más workers que Windows
        use_multiprocessing=False
    )

    # Cargar los mejores pesos antes del fine-tuning
    model.load_weights(CHECKPOINT_PATH)
    print("\n✅ Fase 1 completada. Mejores pesos cargados.")

    print("── FASE 2: Fine-tuning de capas superiores de EfficientNetB3 ──\n")

    base_model.trainable = True

    # Congelar todo excepto las últimas 30 capas
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_f2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Parámetros entrenables en Fase 2: {trainable_f2:,}")

    # LR mucho más bajo para no destruir los pesos preentrenados
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )

    # Nuevo generador MixUp para Fase 2
    train_mixed_f2 = mixup_generator(train_generator, alpha=MIXUP_ALPHA)

    history_fase2 = model.fit(
        train_mixed_f2,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_F2,
        validation_data=val_generator,
        callbacks=get_callbacks(CHECKPOINT_PATH),
        workers=1,
        use_multiprocessing=False
    )

    print("\n✅ Fase 2 completada.")
    # Combinar historial de ambas fases
    combined = {}
    for key in history_fase1.history:
        combined[key] = history_fase1.history[key] + history_fase2.history[key]

    df = pd.DataFrame(combined)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Precisión
    df[['accuracy', 'val_accuracy']].plot(ax=axes[0], color=['steelblue', 'coral'])
    axes[0].axvline(x=len(history_fase1.history['accuracy']) - 1,
                    color='gray', linestyle='--', alpha=0.7, label='Inicio Fase 2')
    axes[0].set_title('Precisión del Modelo', fontsize=13)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Pérdida
    df[['loss', 'val_loss']].plot(ax=axes[1], color=['steelblue', 'coral'])
    axes[1].axvline(x=len(history_fase1.history['loss']) - 1,
                    color='gray', linestyle='--', alpha=0.7, label='Inicio Fase 2')
    axes[1].set_title('Pérdida del Modelo', fontsize=13)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Curvas de Entrenamiento — Fase 1 + Fase 2', fontsize=14, y=1.02)
    plt.tight_layout()

    # Guardar en Drive
    plot_path = os.path.join(DRIVE_BASE, 'curvas_entrenamiento.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Curvas guardadas en: {plot_path}")

    # Resumen final
    best_val_acc = max(combined['val_accuracy'])
    best_epoch   = combined['val_accuracy'].index(best_val_acc) + 1
    print(f"\n{'='*40}")
    print(f"  Mejor val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Alcanzado en época: {best_epoch}")
    print(f"  Modelo guardado en: {CHECKPOINT_PATH}")
    print(f"{'='*40}")

    
    # Evaluar sobre el conjunto de validación completo
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    class_names = list(val_generator.class_indices.keys())

    # Reporte de clasificación
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_title('Matriz de Confusión Normalizada', fontsize=14, pad=15)
    ax.set_ylabel('Clase Real', fontsize=11)
    ax.set_xlabel('Clase Predicha', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(DRIVE_BASE, 'matriz_confusion.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Matriz guardada en: {cm_path}")


if __name__ == '__main__':
    main()