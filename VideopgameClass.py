import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # 1. Configuración de GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configurada y lista.")
        except RuntimeError as e:
            print(e)

    # 2. Configuración de rutas y carpetas
    base_dir = 'videogames-cnn-dataset'
    train_dir = os.path.join(base_dir, 'train')
    
    # Crear carpeta para checkpoints si no existe
    checkpoint_dir = "checkpoints_videojuegos"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 3. Generadores de Datos (Aumento de datos y Validación)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # 20% para validar el mejor modelo
    )

    # Entrenamiento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=256, 
        class_mode='categorical',
        subset='training'
    )

    # Validación (necesario para el checkpoint 'save_best_only')
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='categorical',
        subset='validation'
    )

    # 4. Arquitectura del Modelo
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(21, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 5. Configuración del Checkpoint (Inspirado en el código de Imperial College)
    # Guardará el modelo completo (.h5) solo cuando la precisión de validación mejore
    checkpoint_path = os.path.join(checkpoint_dir, "mejor_modelo.h5")
    
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False, # Guardamos todo el modelo (arquitectura + pesos)
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # 6. Entrenamiento
    print("Iniciando entrenamiento optimizado con Checkpoints...")
    history = model.fit(
        train_generator,
        epochs=15, # Subimos un poco las épocas
        validation_data=val_generator,
        callbacks=[checkpoint_callback],
        workers=8
    )

    # 7. Graficar resultados usando Pandas (como en el ejemplo)
    df = pd.DataFrame(history.history)
    df[['accuracy', 'val_accuracy']].plot(figsize=(10, 5))
    plt.title('Precisión del Modelo')
    plt.savefig('curva_precision.png')
    
    df[['loss', 'val_loss']].plot(figsize=(10, 5))
    plt.title('Pérdida del Modelo')
    plt.savefig('curva_perdida.png')
    
    print(f"\nProceso terminado. El mejor modelo se guardó en: {checkpoint_path}")

if __name__ == '__main__':
    main()
