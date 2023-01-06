from typing import Tuple
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D,\
                                    LeakyReLU, BatchNormalization, Activation,\
                                    Dropout, Concatenate, Lambda, Reshape


class Unet:
    def __init__(self, input_shape: Tuple = (256, 256, 1)) -> None:
        tf.keras.backend.clear_session()
        self.model = self.define_unet(input_shape)

    def __call__(self, x: tf.Tensor):
        return self.model(x)

    def conv_block(self, x: tf.Tensor, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, x: tf.Tensor, skip_features: tf.Tensor, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def define_unet(self, input_shape: Tuple):
        """This is a resnet50_unet"""

        """ Input """
        inputs = Input(input_shape)
        grayscale_input = Lambda(lambda x: x[..., 0][..., tf.newaxis], input_shape=input_shape)(inputs)
        grayscale_resnet_input = Concatenate()([grayscale_input, grayscale_input, grayscale_input])

        """ Pre-trained ResNet50 Model """
        resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=grayscale_resnet_input)

        """ Encoder """
        s1 = resnet50.get_layer("input_1").output           ## (256 x 256)
        s2 = resnet50.get_layer("conv1_relu").output        ## (128 x 128)
        s3 = resnet50.get_layer("conv2_block3_out").output  ## (64 x 64)
        s4 = resnet50.get_layer("conv3_block4_out").output  ## (32 x 32)

        """ Bridge """
        b1 = resnet50.get_layer("conv4_block6_out").output  ## (16 x 16)

        """ Decoder """
        d1 = self.decoder_block(b1, s4, 512)                ## (32 x 32)
        d2 = self.decoder_block(d1, s3, 256)                ## (64 x 64)
        d3 = self.decoder_block(d2, s2, 128)                ## (128 x 128)
        d4 = self.decoder_block(d3, s1, 64)                 ## (256 x 256)

        """ Output """
        outputs = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="ResNet50_U-Net")
        return model


if __name__ == "__main__":
    Unet().model.summary()
