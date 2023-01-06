from typing import Tuple
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D,\
                                    LeakyReLU, BatchNormalization, Activation,\
                                    Dropout, concatenate, UpSampling2D, Lambda, Reshape, RepeatVector


class Fusion:
    def __init__(self, input_shape: Tuple = (256, 256, 1)):
        self.model = self.define_fusion_model(input_shape)

    def __call__(self, x: tf.Tensor, embed: tf.Tensor):
        return self.model([x, embed])

    def define_fusion_model(self, input_shape: Tuple):
        embed_input = Input(shape=(1000,))
        # Encoder
        encoder_input = Input(shape=input_shape)
        encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
        # Fusion
        fusion_output = RepeatVector(32 * 32)(embed_input)
        fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
        fusion_output = concatenate([encoder_output, fusion_output], axis=3)
        fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
        # Decoder
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)


if __name__ == "__main__":
    Fusion().model.summary()
