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
base_dir = 'videogames-cnn-prueba'
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

# (Opcional) Guardar la figura
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')