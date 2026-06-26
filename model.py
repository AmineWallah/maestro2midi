import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import h5py
import numpy as np
from data_pipeline import CHUNK_SIZE

def encoder_block(x, filters):
    # First conv
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Skip connection
    skip = x

    # Pooling to reduce dimensions (halve them with size=2)
    x = MaxPooling2D(pool_size=2)(x)
    return x, skip
def decoder_block(x, skip, filters):
    # Upsample to double the size
    x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)


    # Concatenate with the skip connection
    x = Concatenate()([x, skip])

    # First conv
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x
def build_unet(input_shape=(128, CHUNK_SIZE, 1)):
    inputs = Input(shape=input_shape)
    x = inputs

    # Encoder blocks
    x, skip1 = encoder_block(x, 32)
    x, skip2 = encoder_block(x, 64)
    x, skip3 = encoder_block(x, 128)

    # Bottleneck (just two conv layers)
    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Decoder blocks
    x = decoder_block(x, skip3, 128)
    x = decoder_block(x, skip2, 64)
    x = decoder_block(x, skip1, 32)

    # Output layer
    # x shape at this point: (batch, 128, 156, 32)

    # Move time dimension to front so we can flatten freq × channels per time step
    x = tf.keras.layers.Permute((2, 1, 3))(x)  # → (batch, 156, 128, 32)

    # Flatten frequency and channel dimensions together, shared features both heads see
    shared = tf.keras.layers.Reshape((CHUNK_SIZE, 128 * 32))(x)  # → (batch, 156, 4096)

    # Project down to 88 piano key predictions per time frame
    frame_output = tf.keras.layers.Dense(88, activation='sigmoid', name='frame_output')(shared)
    onset_output = tf.keras.layers.Dense(88, activation='sigmoid', name='onset_output')(shared)
    
    return Model(inputs, [frame_output, onset_output])


def main():
    model = build_unet()
    model.summary()

if __name__ == "__main__":
    main()