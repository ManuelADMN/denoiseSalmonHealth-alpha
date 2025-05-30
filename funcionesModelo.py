import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import tensorflow as tf

# Flag global para detener el entrenamiento en caliente
entrenamiento_activo = False

def cargar_modelos_entrenados(carpeta_modelos="modelos_guardados"):
    """
    Carga modelos previamente entrenados desde una carpeta especificada.

    Requiere que existan los archivos:
    - encoder.h5
    - decoder.h5
    - autoencoder.h5

    Retorna (encoder, decoder, autoencoder) o lanza error si no se encuentran.
    """
    encoder_path     = os.path.join(carpeta_modelos, "encoder.h5")
    decoder_path     = os.path.join(carpeta_modelos, "decoder.h5")
    autoencoder_path = os.path.join(carpeta_modelos, "autoencoder.h5")

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(autoencoder_path)):
        st.error("❌ No se encontraron todos los archivos del modelo en la carpeta seleccionada.")
        st.stop()

    try:
        encoder     = tf.keras.models.load_model(encoder_path)
        decoder     = tf.keras.models.load_model(decoder_path)
        autoencoder = tf.keras.models.load_model(autoencoder_path)
        st.success("✅ Modelos cargados correctamente.")
        return encoder, decoder, autoencoder
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {str(e)}")
        st.stop()

def cargar_datos(dataset_path, target_size=(250, 600), batch_size=32):
    """
    Carga imágenes desde un directorio usando ImageDataGenerator.
    Retorna generadores para entrenamiento y validación.

    Requiere una ruta válida al dataset (no hay fallback).
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        raise ValueError("⚠️ Debes proporcionar una ruta válida al dataset con subcarpetas de clases.")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='input',  # Autoencoder: entrada = salida
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='input',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def entrenar_modelo(autoencoder, x_train, epocas=50, batch_size=32, callback=None):
    """
    Entrena el autoencoder por épocas. Soporta generadores.
    """
    global entrenamiento_activo
    entrenamiento_activo = True
    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epocas):
        if not entrenamiento_activo:
            break

        history = autoencoder.fit(
            x_train,
            validation_data=x_train.validation_data if hasattr(x_train, 'validation_data') else None,
            epochs=1,
            verbose=0
        )

        loss = history.history['loss'][0]
        val_loss = history.history.get('val_loss', [0])[0]
        hist['loss'].append(loss)
        hist['val_loss'].append(val_loss)

        if callback is not None:
            callback(epoch + 1, loss, val_loss)

    return hist

def detener_entrenamiento():
    """
    Detiene el entrenamiento en curso.
    """
    global entrenamiento_activo
    entrenamiento_activo = False



def guardar_modelos(encoder, decoder, autoencoder, carpeta_salida="modelos_guardados"):
    import os
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    encoder.save(os.path.join(carpeta_salida, "encoder.h5"))
    decoder.save(os.path.join(carpeta_salida, "decoder.h5"))
    autoencoder.save(os.path.join(carpeta_salida, "autoencoder.h5"))
    return carpeta_salida


def graficar_entrenamiento(hist):
    """
    Devuelve una figura matplotlib con las curvas de pérdida.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist['loss'], label='Loss entrenamiento')
    ax.plot(hist['val_loss'], label='Loss validación')
    ax.set_title("Curva de pérdida")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    return fig
