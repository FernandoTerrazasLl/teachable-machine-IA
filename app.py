import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Configuración básica de CustomTkinter
ctk.set_appearance_mode("System")  # Soporta "System", "Dark", "Light" -> en macOS se adapta al tema actual
ctk.set_default_color_theme("blue")

class DiagnosticoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Asistente de Diagnóstico Dermatológico")
        self.root.geometry("650x750")
        self.root.minsize(600, 700)

        # Configurar grilla para que los elementos se centren correctamente
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)  # La tarjeta de la imagen tomará el espacio central

        try:
            # Cargar el modelo de IA
            self.model = tf.keras.models.load_model("keras_model1.h5", compile=False)
            with open("labels1.txt", "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Falta el modelo o etiquetas:\n{e}")
            self.root.after(100, self.root.destroy)
            return

        # -----------------------------
        # CABECERA (TÍTULOS)
        # -----------------------------
        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, pady=(40, 10), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.lbl_titulo = ctk.CTkLabel(
            self.header_frame, 
            text="Analizador de Cancer de Piel", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.lbl_titulo.grid(row=0, column=0, pady=(0, 5))

        self.lbl_subtitulo = ctk.CTkLabel(
            self.header_frame, 
            text="Asistente de IA para clasificación dermatológica", 
            font=ctk.CTkFont(size=15),
            text_color="gray"
        )
        self.lbl_subtitulo.grid(row=1, column=0)

        # -----------------------------
        # ÁREA DE LA IMAGEN (TIPO TARJETA)
        # -----------------------------
        self.card_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.card_frame.grid(row=1, column=0, padx=40, pady=20, sticky="nsew")
        self.card_frame.grid_columnconfigure(0, weight=1)
        self.card_frame.grid_rowconfigure(0, weight=1)

        self.mi_imagen = None
        self.lbl_imagen = ctk.CTkLabel(
            self.card_frame, 
            text="\n\nHaz clic en el botón de abajo\npara subir una imagen de la lesión", 
            font=ctk.CTkFont(size=15),
            text_color="gray",
            corner_radius=15,
            width=350,
            height=350,
            fg_color=("gray85", "gray25") # Gris claro en modo Light, Gris oscuro en modo Dark
        )
        self.lbl_imagen.grid(row=0, column=0, padx=20, pady=20)

        # -----------------------------
        # BOTÓN DE ACCIÓN
        # -----------------------------
        self.btn_subir = ctk.CTkButton(
            self.root, 
            text="Cargar Imagen Médica", 
            command=self.cargar_imagen, 
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            corner_radius=25,
            hover_color="#1e6bba"
        )
        self.btn_subir.grid(row=2, column=0, pady=(10, 30))

        # -----------------------------
        # SECCIÓN DE RESULTADOS
        # -----------------------------
        self.resultado_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.resultado_frame.grid(row=3, column=0, pady=(0, 40), sticky="ew")
        self.resultado_frame.grid_columnconfigure(0, weight=1)

        self.lbl_resultado = ctk.CTkLabel(
            self.resultado_frame, 
            text="Esperando imagen...", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray",
            wraplength=600
        )
        self.lbl_resultado.grid(row=0, column=0, pady=(0, 5))

        self.lbl_confianza = ctk.CTkLabel(
            self.resultado_frame, 
            text="Aún no se ha realizado ningún análisis.", 
            font=ctk.CTkFont(size=14),
            text_color="gray",
            wraplength=600
        )
        self.lbl_confianza.grid(row=1, column=0)

        # Diccionario de enfermedades
        self.diccionario_medico = {
            "MEL": "Melanoma (CÁNCER)",
            "BCC": "Carcinoma Basocelular (CÁNCER)",
            "AK": "Queratosis Actínica (CÁNCER)"
        }

    def cargar_imagen(self):
        ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")]
        )
        if ruta_archivo:
            self.mostrar_imagen_ui(ruta_archivo)
            # Retrasar ligeramente la predicción para permitir que la UI alcance a renderizar la foto
            self.root.after(100, lambda: self.predecir_lesion(ruta_archivo))

    def mostrar_imagen_ui(self, ruta):
        img_original = Image.open(ruta)
        
        # Obtener medidas idealmente preservando el aspect ratio original
        img_original.thumbnail((350, 350))
        ancho, alto = img_original.size
        
        # CustomTkinter usa CTkImage para el manejo moderno de imágenes
        self.mi_imagen = ctk.CTkImage(light_image=img_original, dark_image=img_original, size=(ancho, alto))
        self.lbl_imagen.configure(image=self.mi_imagen, text="", fg_color="transparent")
        self.root.update()

    def predecir_lesion(self, ruta):
        self.lbl_resultado.configure(text="⏳ Analizando...", text_color="gray")
        self.lbl_confianza.configure(text="Procesando con la Red Neuronal...")
        self.root.update()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(ruta).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = self.model.predict(data)
        indice_ganador = np.argmax(prediction)
        clase_ganadora = self.class_names[indice_ganador]
        porcentaje = prediction[0][indice_ganador] * 100

        nombre_limpio = clase_ganadora.split(" ", 1)[1] if " " in clase_ganadora else clase_ganadora
        partes = clase_ganadora.split()
        codigo_medico = partes[1] if len(partes) > 1 else nombre_limpio

        diagnostico_completo = self.diccionario_medico.get(codigo_medico, nombre_limpio)

        if codigo_medico in ["MEL", "BCC", "AK"]:
            color = "#ef4444" # Rojo vivo
        else:
            color = "#22c55e" # Verde vivo

        self.lbl_resultado.configure(text=f"{codigo_medico} - {diagnostico_completo}", text_color=color)
        self.lbl_confianza.configure(text=f"Nivel de Confianza: {porcentaje:.1f}%")

        # Audio forzado en español usando la voz Monica
        texto_hablado = f"El diagnóstico indica {diagnostico_completo}. Certeza del {int(porcentaje)} por ciento."
        os.system(f"say -v Monica '{texto_hablado}' &")

if __name__ == "__main__":
    app_root = ctk.CTk()
    app = DiagnosticoApp(app_root)
    app_root.mainloop()