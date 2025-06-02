# funcionesModelo.py

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Flag global para detener el entrenamiento en caliente
entrenamiento_activo = False

def cargar_modelos_entrenados(carpeta_modelos="modelos_guardados"):
    """
    Busca y carga los archivos:
      • encoder_<algo>.h5
      • decoder_<algo>.h5
      • autoencoder_<algo>.h5
    dentro de la carpeta indicada. Si falta alguno o falla la carga, lanza excepción.
    """
    if not os.path.isdir(carpeta_modelos):
        raise FileNotFoundError(f"La carpeta '{carpeta_modelos}' no existe.")

    # Función auxiliar para encontrar primer match de patrón en la carpeta
    def buscar_h5(prefijo):
        for fname in os.listdir(carpeta_modelos):
            if fname.startswith(prefijo) and fname.endswith(".h5"):
                return os.path.join(carpeta_modelos, fname)
        return None

    encoder_path     = buscar_h5("encoder_")
    decoder_path     = buscar_h5("decoder_")
    autoencoder_path = buscar_h5("autoencoder_")

    faltantes = []
    if encoder_path is None:
        faltantes.append("encoder_<algo>.h5")
    if decoder_path is None:
        faltantes.append("decoder_<algo>.h5")
    if autoencoder_path is None:
        faltantes.append("autoencoder_<algo>.h5")

    if faltantes:
        raise FileNotFoundError(
            "No se encontraron los siguientes archivos en la carpeta "
            f"'{carpeta_modelos}': {', '.join(faltantes)}.\n"
            "Debes tener archivos cuyo nombre empiece con:\n"
            "  • encoder_\n"
            "  • decoder_\n"
            "  • autoencoder_\n"
            "Y terminen en .h5"
        )

    st.write(f"Encontrado encoder en: {encoder_path}")
    st.write(f"Encontrado decoder en: {decoder_path}")
    st.write(f"Encontrado autoencoder en: {autoencoder_path}")

    try:
        encoder     = tf.keras.models.load_model(encoder_path, compile=False)
        decoder     = tf.keras.models.load_model(decoder_path, compile=False)
        autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
        st.success("✅ Modelos cargados correctamente.")
        return encoder, decoder, autoencoder
    except Exception as e:
        raise ValueError(
            f"Error al cargar modelos desde '{carpeta_modelos}': {e}\n"
            "Verifica que los archivos sean modelos Keras .h5 válidos."
        )

def cargar_datos(dataset_path, target_size=(128, 128), batch_size=32):
    """
    Carga imágenes desde un directorio usando ImageDataGenerator.
    Ahora usa target_size=(128,128) para que coincida con el autoencoder que espera 128×128×1.

    Retorna dos generadores: train_generator y val_generator.
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        raise ValueError("⚠️ Debes proporcionar una ruta válida al dataset con subcarpetas de clases.")

    # Creamos el ImageDataGenerator con validation_split=0.2
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,       # <-- Aquí cambiamos a (128, 128)
        color_mode='grayscale',        # Asumimos imágenes en escala de grises
        batch_size=batch_size,
        class_mode='input',            # Para autoencoder: entrada = salida
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,       # <-- Igual aquí
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='input',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator


def entrenar_modelo(autoencoder, train_gen, val_gen, epocas, callbacks_list=None, callback_plot=None):
    """
    Entrena el autoencoder por épocas usando los generadores de train y val.
    Llama a callback_plot(epoch, loss, val_loss) al finalizar cada época.
    Permite pasar una lista de callbacks de Keras (por ejemplo lr_scheduler, early_stop).
    """
    global entrenamiento_activo
    entrenamiento_activo = True
    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epocas):
        if not entrenamiento_activo:
            break

        history = autoencoder.fit(
            train_gen,
            validation_data=val_gen,
            epochs=1,
            verbose=0,
            callbacks=callbacks_list or []
        )
        loss     = history.history['loss'][0]
        val_loss = history.history.get('val_loss', [None])[0]
        hist['loss'].append(loss)
        hist['val_loss'].append(val_loss)

        if callback_plot is not None:
            callback_plot(epoch + 1, loss, val_loss)
    return hist


def detener_entrenamiento():
    """
    Detiene el entrenamiento en curso.
    """
    global entrenamiento_activo
    entrenamiento_activo = False


def guardar_modelos(encoder, decoder, autoencoder, carpeta_salida="modelos_guardados"):
    """
    Guarda encoder, decoder y autoencoder en la carpeta indicada
    con extensión .h5 (Single-File).
    Los nombres tendrán el sufijo '_64.h5' para coincidir con el pipeline.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    encoder.save(os.path.join(carpeta_salida, "encoder_64.h5"))
    decoder.save(os.path.join(carpeta_salida, "decoder_64.h5"))
    autoencoder.save(os.path.join(carpeta_salida, "autoencoder_64.h5"))
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


def construir_modelo(input_shape, kernel_size=3, latent_dim=16, layer_filters=[32, 64]):
    """
    Construye y compila un autoencoder convolutional, y también crea callbacks:
      • ReduceLROnPlateau (lr_scheduler)
      • EarlyStopping

    Parámetros:
        input_shape: tupla, forma de entrada (altura, anchura, canales)
        kernel_size: tamaño del kernel para las convoluciones
        latent_dim: dimensión del espacio latente
        layer_filters: lista con la cantidad de filtros por capa

    Devuelve:
        encoder         (tf.keras.Model)
        decoder         (tf.keras.Model)
        autoencoder     (tf.keras.Model, compilado con optimizer y loss)
        callbacks_list  (list de tf.keras.callbacks.Callback)
    """
    # ┏━ ENCODER
    inputs = Input(shape=input_shape, name="encoder_input")
    x = inputs
    for filters in layer_filters:
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            strides=2,
            padding="same"
        )(x)
    shape = K.int_shape(x)  # Guardamos shape para el decoder
    x = Flatten()(x)
    latent = Dense(latent_dim, name="latent_vector")(x)
    encoder = Model(inputs, latent, name="encoder")

    # ┗━ DECODER
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(np.prod(shape[1:]), activation="relu")(decoder_input)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for filters in reversed(layer_filters):
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            strides=2,
            padding="same"
        )(x)
    outputs = Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        activation="sigmoid",
        padding="same",
        name="decoder_output"
    )(x)
    decoder = Model(decoder_input, outputs, name="decoder")

    # ┏━ AUTOENCODER (encoder + decoder)
    autoencoder_input = inputs
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = Model(autoencoder_input, autoencoder_output, name="autoencoder")
    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    # ┏━ CALLBACKS
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    callbacks_list = [lr_scheduler, early_stop]

    return encoder, decoder, autoencoder, callbacks_list
