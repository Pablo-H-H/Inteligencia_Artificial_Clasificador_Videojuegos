import os
import shutil
import random
from pathlib import Path

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
base_dir = "videogames-cnn-dataset_V3"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "validation")   # Nueva carpeta

# Semilla para reproducibilidad
random.seed(42)

# ------------------------------------------------------------
# 1. Unificar: mover todas las imágenes de test a train
# ------------------------------------------------------------
def move_all_from_test_to_train():
    if not os.path.exists(test_dir):
        print(f"La carpeta {test_dir} no existe. Se omite.")
        return
    
    for class_name in os.listdir(test_dir):
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(test_class_dir):
            continue
        
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        
        for filename in os.listdir(test_class_dir):
            src = os.path.join(test_class_dir, filename)
            dst = os.path.join(train_class_dir, filename)
            
            # Evitar sobrescritura
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{base}_copy{counter}{ext}"
                    dst = os.path.join(train_class_dir, new_name)
                    counter += 1
            
            shutil.move(src, dst)
            print(f"Movido: {src} -> {dst}")
        
        # Eliminar subcarpeta vacía de test
        try:
            os.rmdir(test_class_dir)
        except OSError:
            pass
    
    # Eliminar carpeta test si está vacía
    try:
        os.rmdir(test_dir)
    except OSError:
        pass

# ------------------------------------------------------------
# 2. Contar imágenes en una carpeta (por clase)
# ------------------------------------------------------------
def count_images_per_class(directory):
    if not os.path.exists(directory):
        return {}
    counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            num_files = sum(1 for f in os.listdir(class_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')))
            counts[class_name] = num_files
    return counts

# ------------------------------------------------------------
# 3. Mover porcentaje a test y validation (desde train)
# ------------------------------------------------------------
def split_train_into_train_val_test(train_percent=0.8, val_percent=0.1, test_percent=0.1):
    """
    Asume que train_dir contiene todas las imágenes.
    Mueve val_percent a validation y test_percent a test.
    El resto permanece en train.
    """
    # Crear carpetas necesarias
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Para cada clase
    for class_name in os.listdir(train_dir):
        train_class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(train_class_dir):
            continue
        
        # Listar todos los archivos de imagen
        files = [f for f in os.listdir(train_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        total = len(files)
        if total == 0:
            continue
        
        # Calcular cuántos irán a cada destino
        n_val = int(total * val_percent)
        n_test = int(total * test_percent)
        # Asegurar que no se pase del total
        if n_val + n_test > total:
            # Ajustar (por redondeo)
            n_val = max(0, total - n_test)
        
        # Seleccionar aleatoriamente índices
        indices = list(range(total))
        random.shuffle(indices)
        val_indices = indices[:n_val]
        test_indices = indices[n_val:n_val + n_test]
        # El resto se queda en train
        
        # Crear subcarpetas en validation y test
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Mover a validation
        for idx in val_indices:
            filename = files[idx]
            src = os.path.join(train_class_dir, filename)
            dst = os.path.join(val_class_dir, filename)
            # Manejar conflictos de nombre
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{base}_val{counter}{ext}"
                    dst = os.path.join(val_class_dir, new_name)
                    counter += 1
            shutil.move(src, dst)
            print(f"Validation: {src} -> {dst}")
        
        # Mover a test
        for idx in test_indices:
            filename = files[idx]
            src = os.path.join(train_class_dir, filename)
            dst = os.path.join(test_class_dir, filename)
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{base}_test{counter}{ext}"
                    dst = os.path.join(test_class_dir, new_name)
                    counter += 1
            shutil.move(src, dst)
            print(f"Test: {src} -> {dst}")
        
        print(f"Clase '{class_name}': {total} imágenes -> "
              f"train: {total - n_val - n_test}, val: {n_val}, test: {n_test}")

# ------------------------------------------------------------
# 4. Función principal
# ------------------------------------------------------------
def main():
    print("=== REORGANIZACIÓN CON TRAIN/VAL/TEST (80/10/10) ===\n")
    
    # Paso 1: Unificar test -> train
    print("Paso 1: Moviendo todas las imágenes de 'test' a 'train'...")
    move_all_from_test_to_train()
    
    # Paso 2: Contar imágenes unificadas
    print("\nPaso 2: Conteo inicial (todo en 'train'):")
    train_counts = count_images_per_class(train_dir)
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} imágenes")
    
    # Paso 3: Dividir en train, validation, test (80/10/10)
    print("\nPaso 3: Dividiendo en train (80%), validation (10%), test (10%)...")
    split_train_into_train_val_test(train_percent=0.8, val_percent=0.1, test_percent=0.1)
    
    # Paso 4: Mostrar resultados finales
    print("\n=== DISTRIBUCIÓN FINAL ===")
    final_train_counts = count_images_per_class(train_dir)
    final_val_counts = count_images_per_class(val_dir)
    final_test_counts = count_images_per_class(test_dir)
    
    print("Train:")
    for cls, cnt in final_train_counts.items():
        print(f"  {cls}: {cnt}")
    print("Validation:")
    for cls, cnt in final_val_counts.items():
        print(f"  {cls}: {cnt}")
    print("Test:")
    for cls, cnt in final_test_counts.items():
        print(f"  {cls}: {cnt}")



if __name__ == "__main__":
    main()