import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import inception_resnet_v2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# --------------------- CONFIGURACIÓN ---------------------
TEST_DIR = 'videogames-cnn-dataset_V3/test'   # Ruta a la carpeta de test
IMG_SIZE = (299, 299)                         # Tamaño requerido por InceptionResNetV2
BATCH_SIZE = 32                               # Ajusta según tu memoria
MODEL_PATH = 'inception_med_classifier_final' # Directorio del modelo guardado (SavedModel)
# Si guardaste como .h5, cambia a: 'best_phase2.h5'

# --------------------- CARGA DEL MODELO ---------------------
print("Cargando modelo desde:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)
model.summary()

# --------------------- PREPARACIÓN DE DATOS DE PRUEBA ---------------------
# Mismo preprocesamiento que usaste durante el entrenamiento
test_datagen = ImageDataGenerator(
    preprocessing_function=inception_resnet_v2.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # One-hot igual que en training
    shuffle=False               # Importante: no mezclar para obtener etiquetas ordenadas
)

# Mapeo de índices a nombres de clase
class_names = list(test_generator.class_indices.keys())
print("Clases detectadas:", class_names)

# --------------------- EVALUACIÓN BÁSICA ---------------------
print("\nEvaluando modelo en el conjunto de prueba...")
results = model.evaluate(test_generator, verbose=1)

# La salida depende del orden de las métricas al compilar
# Como compilamos con ['accuracy', 'precision', 'recall'], la salida es:
# [loss, accuracy, precision, recall]
loss = results[0]
accuracy = results[1]
precision = results[2]
recall = results[3]

print(f"Pérdida (loss): {loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")

# --------------------- ANÁLISIS DETALLADO (PREDICCIONES) ---------------------
print("\nGenerando predicciones para análisis detallado...")
# Reiniciamos el generador (necesario porque ya se consumió)
test_generator.reset()

# Predecimos todas las imágenes
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)          # Clase predicha (índice)
y_true = test_generator.classes                  # Clase verdadera (índice)

# Classification report de sklearn
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:")
print(cm)

# Visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión - Clasificador de Videojuegos")
plt.tight_layout()
plt.show()

# Opcional: guardar la figura
# plt.savefig('confusion_matrix_test.png')