# app.py

import streamlit as st
from modelo import construir_modelo
from funcionesModelo import (
    cargar_datos,
    entrenar_modelo,
    detener_entrenamiento,
    graficar_entrenamiento,
    cargar_modelos_entrenados,
    guardar_modelos
)
from datasetTransformer import transformar_dataset
import numpy as np
import os
import tensorflow as tf

def pestaña_autoencoder():
    st.title("🧠 Autoencoder Visual")

    # Sidebar: hiperparámetros
    epocas     = st.sidebar.slider("Número de épocas", 1, 100, 50)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)
    latent_dim = st.sidebar.slider("Dimensión latente", 2, 128, 16)

    # Elección de entrenamiento o carga
    st.markdown("### 🔄 ¿Entrenar desde cero o cargar modelo ya entrenado?")
    usar_modelo_existente = st.checkbox("📂 Cargar modelo previamente entrenado")

    autoencoder = None
    encoder = None
    decoder = None
    callbacks_list = None

    # Entrada para carpeta del dataset
    st.markdown("### 📁 Selecciona carpeta con dataset")
    dataset_path = st.text_input(
        "Ruta del dataset con subcarpetas de clases (FreshFish, InfectedFish, etc.)",
        ""
    )

    if not dataset_path.strip():
        st.warning("⚠️ Por favor, ingresa una ruta válida para comenzar.")
        return

    # Intentar cargar datos
    try:
        x_train, x_val = cargar_datos(dataset_path, batch_size=batch_size)
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        return

    # Detectar input_shape a partir de un batch
    try:
        ejemplo_batch = next(x_train)
    except StopIteration:
        st.error("❌ El generador de entrenamiento está vacío.")
        return

    input_shape = ejemplo_batch[0].shape[1:]  # (H, W, C)

    # Si el usuario desea cargar un modelo existente
    if usar_modelo_existente:
        st.markdown("### 🧠 Carga del modelo")
        carpeta_modelo = st.text_input(
            "📂 Ruta de la carpeta del modelo guardado",
            "modelos_guardados"
        )

        if carpeta_modelo.strip():
            try:
                encoder, decoder, autoencoder = cargar_modelos_entrenados(carpeta_modelo)
            except Exception as e:
                st.error(f"❌ Error al cargar modelo: {str(e)}")
                return
        else:
            st.warning("⚠️ Ingresa una ruta válida para el modelo.")
            return

    # Si no usa modelo existente, construye uno nuevo
    if not usar_modelo_existente:
        encoder, decoder, autoencoder, callbacks_list = construir_modelo(
            input_shape=input_shape,
            latent_dim=latent_dim
        )

    # Historial en session_state
    if "hist" not in st.session_state:
        st.session_state.hist = {"loss": [], "val_loss": []}
    if "entrenado" not in st.session_state:
        st.session_state.entrenado = False

    # Contenedor para la gráfica en tiempo real
    chart_container = st.empty()

    def actualizar_grafico(epoch, loss, val_loss):
        st.session_state.hist["loss"].append(loss)
        st.session_state.hist["val_loss"].append(val_loss)
        fig = graficar_entrenamiento(st.session_state.hist)
        chart_container.pyplot(fig)

    # Botones de control (solo si NO se carga modelo existente)
    if not usar_modelo_existente:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Iniciar entrenamiento"):
                # Reiniciar historial
                st.session_state.hist = {"loss": [], "val_loss": []}

                # Llamada a entrenar_modelo con callbacks del modelo y callback de gráfica
                entrenar_modelo(
                    autoencoder,
                    x_train,
                    x_val,
                    epocas=epocas,
                    callbacks_list=callbacks_list,
                    callback_plot=actualizar_grafico,
                )
                st.session_state.entrenado = True

        with col2:
            if st.button("🛑 Detener entrenamiento"):
                detener_entrenamiento()

    # Botón para guardar modelo entrenado
    if not usar_modelo_existente and st.session_state.entrenado:
        st.markdown("### 💾 Guardar modelo entrenado")
        ruta_guardado = st.text_input("📁 Carpeta para guardar modelos", "modelos_guardados")
        if st.button("✅ Guardar modelo"):
            try:
                guardar_modelos(encoder, decoder, autoencoder, carpeta_salida=ruta_guardado)
                st.success(f"✅ Modelos guardados en: `{ruta_guardado}`")
            except Exception as e:
                st.error(f"❌ Error al guardar modelos: {str(e)}")

    # Mostrar reconstrucciones (funciona tanto si cargaste un modelo como si entrenaste uno)
    st.subheader("🔍 Reconstrucciones del modelo")
    col1, col2 = st.columns(2)

    try:
        x_test_batch = next(x_val)
        imgs_originales = x_test_batch[0][:10]
    except Exception:
        st.warning("⚠️ No hay datos de validación para mostrar reconstrucciones.")
        return

    # Llamamos al modelo directamente en modo eager para evitar el error de predict()
    tensor_input    = tf.convert_to_tensor(imgs_originales, dtype=tf.float32)
    decoded_tensor  = autoencoder(tensor_input)
    decoded         = decoded_tensor.numpy()

    # Para mostrar en Streamlit, quitamos el canal extra
    orig_imgs    = [img[:, :, 0] for img in imgs_originales]
    decoded_imgs = [img[:, :, 0] for img in decoded]

    with col1:
        st.write("Imágenes originales")
        st.image(orig_imgs, width=100)
    with col2:
        st.write("Reconstrucciones")
        st.image(decoded_imgs, width=100)


def pestaña_transformacion_dataset():
    st.header("🗂️ Transformador de imágenes a dataset")

    st.markdown("Selecciona una carpeta con subcarpetas de clases que contengan imágenes.")

    carpeta_origen = st.text_input("📁 Ruta de la carpeta origen", "")
    carpeta_destino = st.text_input("📤 Ruta de carpeta destino (opcional)", "")

    if st.button("🚀 Transformar"):
        if carpeta_origen.strip() == "":
            st.error("⚠️ Debes ingresar una ruta válida.")
        else:
            try:
                total = transformar_dataset(
                    carpeta_origen,
                    carpeta_destino or "dataset_transformado"
                )
                st.success(f"✅ Dataset creado exitosamente con {total} imágenes procesadas.")
            except Exception as e:
                st.error(f"❌ Error al procesar imágenes: {str(e)}")


def main():
    st.set_page_config(layout="wide")
    tab_auto, tab_dataset = st.tabs(["🧠 Autoencoder", "🗂️ Convertidor Dataset"])

    with tab_auto:
        pestaña_autoencoder()
    with tab_dataset:
        pestaña_transformacion_dataset()


if __name__ == "__main__":
    main()
