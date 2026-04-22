import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = 'videogames-cnn-dataset_V3'
train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
							rescale = 1./255,
							rotation_range = 10,
							width_shift_range = 0.2,
						#	height_shift_range = 0.2,
						#	shear_range = 0.3,
							zoom_range = 0.3,
							horizontal_flip = True)

# %%
train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (300, 300),
							batch_size = 400,
							class_mode ='categorical',
							)

path = "videogames-cnn-dataset"

train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (300, 300),
							batch_size = 400,
							class_mode ='categorical',
							#save_to_dir= path + '/augmented',
							save_prefix='aug',
							save_format='png'
							)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    val_dir, # Usa tu carpeta de validation
    target_size=(300, 300),
    batch_size=32, # Batch pequeño para validación es suficiente
    class_mode='categorical'
)


# %%
# --- AQUÍ LA IMPLEMENTACIÓN DE PREFETCH ---
# Convertimos el generador manual a un Dataset de TensorFlow
def train_gen_callable():
    for batch in train_generator:
        yield batch

train_dataset = tf.data.Dataset.from_generator(
    train_gen_callable,
    output_signature=(
        tf.TensorSpec(shape=(None, 300, 300, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 6), dtype=tf.float32)
    )
)

# Aplicamos prefetch (AUTOTUNE decide cuántos batches adelantar según tu RAM)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# 1. Activamos mixed precision para aprovechar los Tensor Cores de tu RTX 3060
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

model = models.Sequential()

# Capas intermedias (usarán float16 automáticamente)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloque 4: Un bloque extra para profundidad
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))

# 2. Capa de salida: Es fundamental que sea float32 para evitar errores de precisión
# Nota: Usas sigmoid y binary_crossentropy para 3 clases; 
# esto sugiere que es un problema multi-etiqueta.
model.add(layers.Dense(6, activation='softmax', dtype='float32'))

model.summary()

model.compile(loss='categorical_crossentropy',
						optimizer=optimizers.Adam(learning_rate=1e-3),
						metrics=['acc'])


# %%
import tensorflow as tf

print(tf.__version__)

# Listar las GPUs disponibles
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", len(gpus))

if gpus:
    for gpu in gpus:
        print(f"Nombre del dispositivo: {gpu}")
else:
    print("No se detectó ninguna GPU. Se usará la CPU.")

# %%
import tensorflow as tf

# Esto debe devolver una lista con tu GPU
print("Dispositivos detectados:", tf.config.list_physical_devices())

# Esto confirma que TF puede usar la GPU para cálculos
print("¿TensorFlow puede usar la GPU?:", tf.test.is_built_with_cuda())


# %%
from tensorflow.keras.callbacks import ModelCheckpoint

# %%
# Create Tensorflow checkpoint object with epoch and batch details
checkpoint_path = "/checkpoints_Base/checkpoint.weights.h5"
checkpoint_50 = ModelCheckpoint(filepath = checkpoint_path,
                                  save_weights_only = False,
                                  save_freq = "epoch",
                                  monitor = "val_acc",
                                  save_best_only = True,                                  
                                  verbose = 1)

# %%

history = model.fit(
    train_dataset, # <--- Usamos el dataset con prefetch
    steps_per_epoch=len(train_generator), # Obligatorio al usar datasets infinitos
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_50]
)


acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs,acc,'bo',label='train accuracy')
plt.title('train acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss, 'bo', label ='training loss')
plt.title('train loss')
plt.legend()

plt.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Configuración de rutas y parámetros
# ------------------------------------------------------------
base_dir = 'videogames-cnn-dataset_V3'
test_dir = os.path.join(base_dir, 'test')

checkpoint_path_best = "/checkpoints_Base/checkpoint.weights.h5"

batch_size = 32
target_size = (300, 300)
class_mode = 'categorical'

# ------------------------------------------------------------
# 1. Cargar el modelo
# ------------------------------------------------------------
print(f"Cargando modelo desde: {checkpoint_path_best}")
new_model = tf.keras.models.load_model(checkpoint_path_best)
print("Modelo cargado exitosamente.\n")

# ------------------------------------------------------------
# 2. Crear el generador de prueba (mismo preprocesamiento)
# ------------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=False,            # Importante: mantener orden para comparar etiquetas
    # El generador asigna clases en orden alfabético de nombres de carpetas
)

# Guardar nombres de clases (en el orden usado por el generador)
class_names = list(test_generator.class_indices.keys())
print("Clases:", class_names)

# ------------------------------------------------------------
# 3. Evaluación global (pérdida y precisión)
# ------------------------------------------------------------
loss, accuracy = new_model.evaluate(test_generator, verbose=1)
print(f"\nPérdida en prueba: {loss:.4f}")
print(f"Precisión (accuracy) en prueba: {accuracy:.4f}\n")

# ------------------------------------------------------------
# 4. Obtener predicciones y etiquetas verdaderas
# ------------------------------------------------------------
# Reiniciamos el generador para asegurar que empezamos desde el primer lote
test_generator.reset()

# Predecir sobre todo el conjunto de prueba
pred_probabilities = new_model.predict(test_generator, verbose=1)

# Convertir probabilidades a etiquetas de clase (índice de la mayor probabilidad)
pred_classes = np.argmax(pred_probabilities, axis=1)

# Obtener las etiquetas verdaderas como enteros
true_classes = test_generator.classes   # Porque shuffle=False

# ------------------------------------------------------------
# 5. Matriz de confusión y métricas por clase
# ------------------------------------------------------------
cm = confusion_matrix(true_classes, pred_classes)

# Reporte de clasificación (precisión, recall, f1-score por clase)
print("=== Reporte de clasificación ===")
print(classification_report(true_classes, pred_classes, target_names=class_names))

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión - Conjunto de Prueba')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.show()



