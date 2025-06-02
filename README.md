# 🧠 DenoiseSH 0.2.0-alpha (DenoiseSalmonHealth) - Autoencoder para imágenes de salmón

Este proyecto implementa una app visual interactiva en Streamlit para entrenar un **autoencoder convolucional** que detecta y reconstruye imágenes de salmón saludables e infectados, ideal para experimentación con **biotecnología y salud acuícola**.

---

## 🧠 Conocimientos utilizados

* **Lenguaje de programación:** Python 3.9 / 3.10
* **Frameworks y librerías de ML/DL:** TensorFlow 2.11, Keras
* **Procesamiento de imágenes:** NumPy, Pandas, Matplotlib, Scikit-Image, OpenCV
* **Interfaz web interactiva:** Streamlit
* **Técnicas de redes neuronales:** Autoencoders convolucionales, Data Augmentation, Callbacks
* **GPU y aceleración:** CUDA Toolkit 11.2, cuDNN 8.1
* **Control de versiones y colaboración:** Git, GitHub

---

## ⚙️ Requisitos

* Python **3.9 o 3.10**
* GPU compatible con CUDA (opcional, pero recomendado)
* pip

---

## 🧪 Crear entorno virtual y activar

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En MacOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Instalar dependencias

```bash
pip install -r requirements.txt
```

> Asegúrate de tener instalados los drivers y librerías CUDA/cuDNN si quieres usar la **GPU** (ver más abajo).

---

## 🧠 Ejecutar la app

```bash
streamlit run app.py
```

Esto abrirá tu navegador por defecto con la interfaz Streamlit.

---

## 📁 Estructura esperada del dataset

La carpeta debe contener subcarpetas por clase, por ejemplo:

```
SalmonScan/
├── FreshFish/
│   ├── img1.png
│   ├── img2.png
├── InfectedFish/
│   ├── img3.png
│   ├── img4.png
```

Puedes seleccionar esta carpeta desde la interfaz para procesarla o entrenar con ella.

---

## 💾 Estructura del proyecto

```
denoiseSalmonHealth/
├── app.py                      # Interfaz principal Streamlit
├── modelo.py                  # Arquitectura del Autoencoder
├── funcionesModelo.py         # Entrenamiento, gráficas, utilidades
├── datasetTransformer.py      # Transformador visual de datasets
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

---

## ⚡ Activar soporte GPU (opcional pero recomendado)

Para que TensorFlow use la **GPU**, debes tener instalados:

| Componente    | Versión   |
| ------------- | --------- |
| TensorFlow    | 2.11.0    |
| CUDA Toolkit  | 11.2      |
| cuDNN         | 8.1       |
| Driver NVIDIA | >= 460.xx |

* Puedes verificar si tu GPU está activa con:

```python
# En un script o consola Python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

---

## ❗ Problemas comunes

* **`NameError: name 'scipy' is not defined`**
  → Asegúrate de que `scipy` esté en `requirements.txt`.

* **No carga el dataset desde `.zip`**
  → Extrae el `.zip` antes de seleccionarlo. No se puede leer directo desde el archivo comprimido.

* **Dataset no encontrado**
  → Usa la interfaz para seleccionar carpetas completas, no imágenes sueltas.

---

## ❤️ Desarrollado por Manuel Díaz

Proyecto creado con amor por un estudiante de Ingeniería en Informática amante de la IA y la innovación.

---

Este repositorio es de solo lectura; para usos o contribuciones, contáctame.
