
# ⚓ DenoiseSH v0.5-alpha – Diagnóstico visual de salmones con IA

Interfaz minimalista y ultra intuitiva desarrollada en Streamlit para detectar automáticamente la salud de salmones mediante redes neuronales y clasificación robusta. Versión actualizada con nuevas funcionalidades, mayor eficacia y una mejor experiencia de usuario.

---

## 🎯 Novedades en la versión 0.5-alpha

- **Dashboard optimizado:** Nueva estructura con pestañas claras para carga de modelos, exploración, inferencia y evaluación.
- **Carga directa de modelos:** Ahora puedes subir directamente desde la interfaz tus modelos `encoder.keras` y `classifier.joblib`, o incluso en formato ZIP.
- **Predicción estocástica mejorada:** Hasta **50 iteraciones** para predicciones robustas con dropout activo.
- **Aumento significativo de accuracy:** Ajuste dinámico del umbral de decisión directamente desde la interfaz.
- **Evaluación continua:** Nuevo loop automático que permite pruebas continuas mostrando métricas en tiempo real.


## 🔧 Gestión y exportación de modelos

**Importante:** Los modelos no se exportan directamente desde la app por seguridad y consistencia. Deben ser generados externamente con los siguientes comandos:

- **Encoder (Keras)**:
```python
encoder.save("encoder.keras")
```
* **Clasificador (Scikit-learn)**:

```python
import joblib
joblib.dump(clasificador, "classifier.joblib")
```

Coloca estos modelos en la carpeta `models/` o súbelos mediante la interfaz para su uso automático.

---

## 📂 Estructura esperada del dataset

Asegúrate de organizar tus imágenes en la carpeta `dataset/` con esta estructura:

```
dataset/
├── FreshSalmon/
│   ├── img001.png
│   └── ...
├── InfectedSalmon/
│   ├── img123.png
│   └── ...
```

---

## ⚙️ Requisitos del sistema

* Python 3.9 o 3.10
* Streamlit ≥ 1.20
* TensorFlow ≥ 2.9
* Joblib, NumPy, Pillow
* GPU compatible con CUDA (opcional, recomendado)

---

## 🚀 Instrucciones rápidas

### 1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar aplicación

```bash
streamlit run app.py
```

---

## 📁 Estructura del proyecto actualizada (v0.5-alpha)

```
denoiseSalmonHealth/
├── app.py                   # Interfaz Streamlit principal
├── modelsFunction.py        # Lógica y carga de modelos
├── models/                  # Almacén de modelos exportados
├── dataset/                 # Imágenes clasificadas para inferencia
├── requirements.txt         # Dependencias necesarias
└── README.md                # Este archivo
```

---

## ⚡ ¿Cómo verificar la GPU?

Si tienes GPU compatible, verifica que TensorFlow la detecte:

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

---

## ❗ Problemas comunes y soluciones rápidas

* **No carga modelos:** Asegúrate que `encoder.keras` y `classifier.joblib` existan en la carpeta `models/`.
* **Error en predicción:** Revisa que las imágenes tengan las dimensiones correctas `(250, 600)`.

---

## 📊 Resultados obtenidos y mejoras

Con la implementación de la **predicción estocástica**, ajustes del **umbral dinámico** y manejo más eficiente de modelos, hemos conseguido aumentar la precisión (accuracy) hasta en un **15%** comparado con versiones anteriores.

Esto es especialmente útil en datasets desbalanceados donde la precisión es crítica.

---

## ❤️ Acerca del proyecto

Creado por **Manuel Díaz**, estudiante de Ingeniería en Informática en Duoc UC Puerto Montt, entusiasta de la IA y la innovación tecnológica.


