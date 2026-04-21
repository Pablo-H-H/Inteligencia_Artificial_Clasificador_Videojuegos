import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = 'videogames-cnn-dataset_V2'
train_dir = os.path.join(base_dir,'train')
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
    test_dir, # Usa tu carpeta de test
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
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
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
# Bloque 1: Detecta bordes y colores básicos
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Bloque 2: Detecta formas más complejas (armas, personajes)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloque 3: Detecta patrones abstractos de géneros
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
model.add(layers.Dense(3, activation='softmax', dtype='float32'))

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



