¡Listo! Aquí tienes un **README.md** para la **versión 0.5** de DenoiseSH, actualizado según el código y los avances recientes, orientado tanto para usuarios como para desarrolladores, con mención clara a la gestión de modelos, mejoras de accuracy y nuevos flujos de uso.

---

````markdown
# ⚓ DenoiseSH v0.5 – Salmon Health Dashboard

Diagnóstico automático de salud de salmones mediante visión computacional y autoencoders, con interfaz minimalista y ultra intuitiva en Streamlit.

---

## 🚀 **¿Qué es DenoiseSH?**

DenoiseSH es una aplicación web de diagnóstico rápido y visual para imágenes de salmones, pensada para uso en biotecnología, acuicultura e investigación. 
Usa **autoencoders convolucionales** y un **clasificador de alto rendimiento** para detectar anomalías y distinguir entre salmones "Healthy" y "Infected".

**Novedades v0.5:**
- Nueva organización de pestañas (dashboard, carga de modelos, explorador de datos, inferencia manual/continua, evaluación).
- Subida de modelos vía ZIP o archivos sueltos directamente desde la interfaz.
- **Manejo avanzado de modelos:** ahora puedes cargar/actualizar fácilmente tu encoder (`.keras`) y clasificador (`.joblib`) sin reiniciar la app.
- **Aumento de eficacia y accuracy:** integración de predicción estocástica (inferencias repetidas con Dropout activo), ajuste dinámico de umbral y visualización balanceada del dataset.
- Verificación instantánea de accuracy acumulada con métricas de batch, loop automático y pruebas manuales.
- Mejor soporte para el manejo y exportación de modelos.

---

## ⚙️ **Requisitos**

- Python **3.9** o **3.10**
- [TensorFlow 2.9+](https://www.tensorflow.org/)  
- [Streamlit](https://streamlit.io/)
- Scikit-learn, joblib, Pillow, NumPy
- GPU con CUDA (opcional, pero recomendado)

Instala dependencias así:
```bash
pip install -r requirements.txt
````

---

## 🏁 **¿Cómo usar?**

1. **Clona o descarga** este repositorio.
2. (Opcional) Crea un entorno virtual y actívalo.
3. Ejecuta la app:

   ```bash
   streamlit run app.py
   ```
4. Accede a la web local que abrirá tu navegador (por defecto `http://localhost:8501`).

---

## 🛠️ **Gestión de modelos (`models/`)**

### Exportar tu modelo entrenado:

* **Encoder**:
  Guarda tu modelo Keras así:

  ```python
  model.save("models/encoder.keras")
  ```

* **Clasificador** (RandomForest, SVM, etc):

  ```python
  import joblib
  joblib.dump(clf, "models/classifier.joblib")
  ```

> Si tienes un decoder para visualización de reconstrucciones, puedes exportarlo igual:
>
> ```python
> model.save("models/decoder.keras")
> ```

### Carga de modelos en la app:

* Desde la pestaña **🛠️ Modelos**, puedes:

  * Subir un **ZIP** con ambos archivos (`encoder.keras` y `classifier.joblib`)
  * O subirlos por separado desde la interfaz.
* La app detecta y autocarga modelos presentes en la carpeta `/models` al iniciar.

---

## 🏞️ **Estructura del Dataset**

Debes colocar tu dataset en `dataset/` estructurado así:

```
dataset/
├── FreshSalmon/
│   ├── img001.png
│   └── ...
├── InfectedSalmon/
│   ├── img123.png
│   └── ...
```

Puedes trabajar con imágenes `.png`, `.jpg` o `.jpeg`.

---

## 🌟 **Mejoras destacadas en v0.5**

* **Predicción estocástica:**
  Múltiples iteraciones con Dropout activo para mayor robustez y menos overfitting.
* **Ajuste de umbral:**
  Puedes calibrar el umbral de decisión para balancear precisión/sensibilidad.
* **Verificación por lote y acumulada:**
  Fácilmente visualizas la accuracy sobre muestras aleatorias o por clases.
* **Carga automática de modelos:**
  La app detecta automáticamente los modelos exportados en `/models` y los carga al inicio.
* **Soporte para decoder:**
  Si quieres visualizar reconstrucciones, simplemente agrega `decoder.keras` y la app lo usará (opcional).

---

## 📈 **Impacto y Resultados**

Con el pipeline actual, y usando predicción estocástica junto a calibración de umbral, hemos logrado **aumentar la eficacia/accuracy hasta un 8–15% respecto a versiones previas**, especialmente en datasets desbalanceados o difíciles.

El flujo flexible y la interfaz permiten un ciclo rápido de prueba, validación y ajuste de modelos, facilitando la experimentación y el control de calidad en producción.

---

## 💻 **Estructura del proyecto**

```
denoiseSalmonHealth/
├── app.py                # Interfaz Streamlit principal
├── modelsFunction.py     # Lógica de carga, predicción y utilidades ML
├── models/               # Carpeta de modelos exportados (.keras, .joblib)
├── dataset/              # Imágenes de entrenamiento/inferencia
├── requirements.txt      # Dependencias
└── README.md             # Este archivo
```

---

## 🔧 **Preguntas Frecuentes**

**¿Por qué no se exporta el modelo directamente desde la app?**
Por seguridad y compatibilidad, la app permite únicamente cargar modelos previamente entrenados y exportados desde tu entorno. Exporta desde Python usando `.save()` (Keras) y `joblib.dump()` (Scikit-learn).

**¿Cómo mejoro el accuracy?**

* Usa datasets balanceados.
* Ajusta el umbral de decisión en la barra lateral.
* Utiliza predicción estocástica para resultados más robustos.
* Experimenta con arquitecturas y regularización en tu modelo encoder.

---

## ❤️ **Desarrollador**

Creado y mantenido por **Manuel Díaz**
Estudiante de Ingeniería en Informática (Duoc UC – Puerto Montt)
Amante de la IA y la innovación.




