import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Intentar importar soporte para arrastrar y soltar (opcional)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False
    TkinterDnD = tk.Tk  # fallback a Tk normal

# ------------------------------------------------------------
# Configuración de rutas y parámetros
# ------------------------------------------------------------
BASE_DIR = 'videogames-cnn-prueba'
TEST_DIR = os.path.join(BASE_DIR, 'test')

# *** CAMBIO 1: Ruta del nuevo modelo (SavedModel o .h5) ***
MODEL_PATH = 'best_phase2.h5'   # carpeta del SavedModel
# Si guardaste como .h5 usa: 'best_phase2.h5'

# *** CAMBIO 2: Tamaño requerido por InceptionResNetV2 ***
IMG_SIZE = (299, 299)

# ------------------------------------------------------------
# 1. Cargar el modelo (si existe)
# ------------------------------------------------------------
model = None
class_names = []

def cargar_modelo():
    global model, class_names
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"No se encontró el modelo en:\n{MODEL_PATH}")
        return False
    
    try:
        # load_model funciona tanto con carpeta SavedModel como con archivo .h5
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo cargado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el modelo:\n{e}")
        return False
    
    # Obtener nombres de clases a partir de las subcarpetas de test (si existen)
    if os.path.exists(TEST_DIR):
        class_names = sorted([d for d in os.listdir(TEST_DIR) 
                              if os.path.isdir(os.path.join(TEST_DIR, d))])
        if class_names:
            print(f"Clases detectadas: {class_names}")
        else:
            messagebox.showwarning("Advertencia", "No se encontraron subcarpetas en 'test'. Las clases se obtendrán del modelo (índices numéricos).")
    else:
        messagebox.showwarning("Advertencia", f"No existe la carpeta {TEST_DIR}. Se usarán índices numéricos como clases.")
    
    return True

# ------------------------------------------------------------
# Función para predecir una imagen dada su ruta
# ------------------------------------------------------------
def predecir_imagen(ruta_imagen):
    if model is None:
        return None, "Modelo no cargado."
    
    try:
        # Cargar y preprocesar imagen
        img = load_img(ruta_imagen, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        
        # *** CAMBIO 3: Preprocesamiento específico de InceptionResNetV2 ***
        # (normaliza a [-1, 1] igual que en entrenamiento)
        img_array = preprocess_input(img_array)
        
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predecir
        predicciones = model.predict(img_batch, verbose=0)[0]
        idx_pred = np.argmax(predicciones)
        confianza = predicciones[idx_pred]
        
        # Nombre de la clase
        if class_names and idx_pred < len(class_names):
            clase = class_names[idx_pred]
        else:
            clase = f"Clase {idx_pred}"
        
        return clase, confianza
    except Exception as e:
        return None, f"Error en predicción: {e}"

# ------------------------------------------------------------
# Función para copiar imagen a la carpeta test/{clase}
# ------------------------------------------------------------
def copiar_a_test(ruta_origen, clase_destino):
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    carpeta_clase = os.path.join(TEST_DIR, clase_destino)
    os.makedirs(carpeta_clase, exist_ok=True)
    
    # Generar nombre único basado en timestamp
    nombre_archivo = os.path.basename(ruta_origen)
    nombre_base, extension = os.path.splitext(nombre_archivo)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nuevo_nombre = f"{nombre_base}_{timestamp}{extension}"
    ruta_destino = os.path.join(carpeta_clase, nuevo_nombre)
    
    try:
        shutil.copy2(ruta_origen, ruta_destino)
        return True, ruta_destino
    except Exception as e:
        return False, str(e)

# ------------------------------------------------------------
# Interfaz gráfica con tkinter
# ------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Videojuegos - InceptionResNetV2")
        self.root.geometry("700x600")
        
        # Variable para ruta de imagen actual
        self.ruta_actual = None
        self.imagen_tk = None  # para mostrar en GUI
        
        # Verificar si el modelo se cargó
        if not cargar_modelo():
            self.root.destroy()
            return
        
        # Configurar interfaz
        self.crear_widgets()
        
        # Configurar drag & drop si está disponible
        if DRAG_DROP_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
    
    def crear_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        titulo = ttk.Label(main_frame, text="Arrastra una imagen o escribe su ruta", 
                           font=('Helvetica', 14))
        titulo.pack(pady=10)
        
        # Área para mostrar imagen (canvas con scroll)
        canvas_frame = ttk.LabelFrame(main_frame, text="Vista previa", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#f0f0f0', height=250)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame para entrada de ruta
        ruta_frame = ttk.Frame(main_frame)
        ruta_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(ruta_frame, text="Ruta de imagen:").pack(side=tk.LEFT)
        self.entry_ruta = ttk.Entry(ruta_frame, width=50)
        self.entry_ruta.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.entry_ruta.bind('<Return>', lambda e: self.predecir_desde_entry())
        
        ttk.Button(ruta_frame, text="Examinar", command=self.seleccionar_archivo).pack(side=tk.LEFT, padx=5)
        
        # Botón predecir
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="🔮 Predecir", command=self.predecir_desde_entry).pack(side=tk.LEFT, padx=5)
        
        # Resultado de predicción
        result_frame = ttk.LabelFrame(main_frame, text="Resultado", padding=10)
        result_frame.pack(fill=tk.X, pady=10)
        
        self.lbl_resultado = ttk.Label(result_frame, text="Esperando imagen...", font=('Helvetica', 12))
        self.lbl_resultado.pack()
        
        self.lbl_confianza = ttk.Label(result_frame, text="", font=('Helvetica', 10))
        self.lbl_confianza.pack()
        
        # Frame para guardar en test
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(save_frame, text="Guardar en test como clase:").pack(side=tk.LEFT)
        
        # Combobox con las clases disponibles (o entrada libre)
        self.combo_clase = ttk.Combobox(save_frame, values=class_names, state="readonly", width=20)
        if class_names:
            self.combo_clase.set(class_names[0])
        else:
            self.combo_clase.set("")
        self.combo_clase.pack(side=tk.LEFT, padx=5)
        
        self.btn_guardar = ttk.Button(save_frame, text="💾 Copiar a carpeta test", 
                                      command=self.guardar_en_test, state=tk.DISABLED)
        self.btn_guardar.pack(side=tk.LEFT, padx=5)
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10,0))
        
        # Configurar redimensionamiento
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def on_drop(self, event):
        """Maneja el evento de soltar archivos arrastrados."""
        archivos = self.root.tk.splitlist(event.data)
        if archivos:
            ruta = archivos[0]  # tomar el primero
            self.cargar_imagen(ruta)
    
    def seleccionar_archivo(self):
        """Abre diálogo para seleccionar archivo de imagen."""
        tipos = [("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"), ("Todos", "*.*")]
        ruta = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=tipos)
        if ruta:
            self.cargar_imagen(ruta)
    
    def cargar_imagen(self, ruta):
        """Carga y muestra la imagen en el canvas."""
        try:
            img = Image.open(ruta)
            # Redimensionar manteniendo proporción para vista previa
            img.thumbnail((500, 250))
            self.imagen_tk = ImageTk.PhotoImage(img)
            
            # Limpiar canvas y mostrar
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                                     anchor=tk.CENTER, image=self.imagen_tk)
            
            self.ruta_actual = ruta
            self.entry_ruta.delete(0, tk.END)
            self.entry_ruta.insert(0, ruta)
            self.status_var.set(f"Imagen cargada: {os.path.basename(ruta)}")
            
            # Habilitar botón guardar si hay modelo
            self.btn_guardar.config(state=tk.NORMAL)
            
            # Limpiar resultado anterior
            self.lbl_resultado.config(text="Imagen cargada. Haz clic en 'Predecir'.")
            self.lbl_confianza.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")
            self.status_var.set("Error al cargar imagen")
    
    def predecir_desde_entry(self):
        """Toma la ruta del entry y ejecuta predicción."""
        ruta = self.entry_ruta.get().strip()
        if not ruta:
            messagebox.showwarning("Atención", "Por favor ingresa una ruta o selecciona una imagen.")
            return
        
        if not os.path.exists(ruta):
            messagebox.showerror("Error", f"El archivo no existe:\n{ruta}")
            return
        
        # Cargar imagen en canvas si no está ya
        if self.ruta_actual != ruta:
            self.cargar_imagen(ruta)
        
        # Ejecutar predicción
        self.status_var.set("Prediciendo...")
        self.root.update()
        
        clase, confianza = predecir_imagen(ruta)
        
        if clase is None:
            messagebox.showerror("Error", confianza)
            self.status_var.set("Error en predicción")
            return
        
        # Mostrar resultado
        self.lbl_resultado.config(text=f"Predicción: {clase}")
        self.lbl_confianza.config(text=f"Confianza: {confianza:.2%}")
        
        # Seleccionar la clase predicha en el combobox
        if class_names and clase in class_names:
            self.combo_clase.set(clase)
        elif class_names:
            # Si la clase no está en la lista (ej. "Clase 0"), dejamos el primero
            pass
        
        self.status_var.set(f"Predicción completada: {clase}")
        self.btn_guardar.config(state=tk.NORMAL)
    
    def guardar_en_test(self):
        """Copia la imagen actual a la carpeta test/{clase_seleccionada}."""
        if not self.ruta_actual:
            messagebox.showwarning("Atención", "No hay imagen cargada.")
            return
        
        clase_destino = self.combo_clase.get()
        if not clase_destino:
            messagebox.showwarning("Atención", "Selecciona o escribe una clase de destino.")
            return
        
        exito, mensaje = copiar_a_test(self.ruta_actual, clase_destino)
        if exito:
            messagebox.showinfo("Éxito", f"Imagen copiada a:\n{mensaje}")
            self.status_var.set(f"Copiada a test/{clase_destino}")
        else:
            messagebox.showerror("Error", f"No se pudo copiar la imagen:\n{mensaje}")

# ------------------------------------------------------------
# Punto de entrada principal
# ------------------------------------------------------------
if __name__ == "__main__":
    if DRAG_DROP_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("Nota: Instala 'tkinterdnd2' para habilitar arrastrar y soltar.")
    
    app = App(root)
    root.mainloop()