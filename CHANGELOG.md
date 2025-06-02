
# 🧾 Changelog — `denoiseSalmonHealth v0.2.0-alpha`
**Fecha de lanzamiento:** 2025-06-01

---

## 🔧 Cambios Principales
- **Ajuste de la arquitectura del Autoencoder**:  
  Rediseñada la red para aceptar imágenes de entrada con forma `(250, 600, 1)` en lugar de la forma fija `(128, 128, 1)`.  
  Esto permite trabajar con las dimensiones reales de las imágenes sin necesidad de interpolación previa, preservando mejor la información relevante para la salud del salmón.

- **Preprocesamiento adaptativo de imágenes**:  
  El flujo de preprocesamiento ahora ajusta dinámicamente el tamaño de entrada según los requerimientos del modelo o aplica `resize()` si es necesario.

- **Soporte para lote dinámico (`batch_size`)**:  
  Añadida capacidad para trabajar con distintos tamaños de lotes sin necesidad de rehacer el grafo de la red.  
  Esto soluciona errores como:  
  `ValueError: Input 0 of layer "autoencoder" is incompatible with the layer`.

---

## 🧪 Mejoras y Estabilidad
- Validaciones de forma (`tensor.shape`) antes de alimentar el modelo.
- Añadido soporte para visualización previa de las dimensiones del tensor cargado vía Streamlit.
- Mejor manejo de errores para cargas de imágenes no compatibles.
- Documentación y mensajes de error más descriptivos durante la inferencia con el autoencoder.

---

## 🧱 Infraestructura y Técnicas
- Modularización del código de red y preprocesamiento.
- Preparación del entorno para entrenamiento incremental en versiones futuras.

---

## ⚠️ Notas
- Esta es una versión **alpha**, orientada a pruebas internas. El modelo aún no ha sido entrenado con datos etiquetados reales.
- Las métricas de reconstrucción no están disponibles en esta versión. Serán agregadas en `v0.3.x`.
