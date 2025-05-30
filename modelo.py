import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def construir_modelo(input_shape,
                     kernel_size=3,
                     latent_dim=16,
                     layer_filters=[32, 64]):
    """
    Construye y compila un autoencoder convolutional.
    Parámetros:
        input_shape: tupla, forma de entrada (altura, anchura, canales)
        kernel_size: tamaño del kernel para las convoluciones
        latent_dim: dimensión del espacio latente
        layer_filters: lista con la cantidad de filtros por capa
    Devuelve:
        encoder, decoder, autoencoder (modelos Keras)
    """
    # ┏━ ENCODER
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    shape = K.int_shape(x)  # Guardamos shape para el decoder
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    encoder = Model(inputs, latent, name='encoder')

    # ┗━ DECODER
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(np.prod(shape[1:]), activation='relu')(decoder_input)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for filters in reversed(layer_filters):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    decoder = Model(decoder_input, outputs, name='decoder')

    # ┏━ AUTOENCODER (encoder + decoder)
    autoencoder_input = inputs
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = Model(autoencoder_input, autoencoder_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    return encoder, decoder, autoencoder
