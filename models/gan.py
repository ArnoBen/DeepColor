from typing import Tuple
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization,\
                                    Activation, Concatenate
from unet import Unet


class GAN():
    def __init__(self, image_shape=(256,256,3), grayscale_shape=(256, 256, 1)) -> None:
        self.image_shape = image_shape
        self.grayscale_shape = grayscale_shape
        self.generator = self.make_generator(grayscale_shape)
        self.discriminator = self.make_discriminator(image_shape, grayscale_shape)
        self.gan = self.make_gan()
        
    def __call__(self):
        raise NotImplementedError
        
    @staticmethod
    def make_generator(grayscale_shape):
        return Unet(grayscale_shape)

    @staticmethod
    def make_discriminator(image_shape, grayscale_shape):
        """
        INPUT 1 (src): L (grayscale) single channel (h, w, 1)
        INPUT 2 (target): Lab colored image (h, w, 3)
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=grayscale_shape)
        # target image input
        in_target_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model

    def make_gan(self):
        # make weights in the discriminator not trainable
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        # define the source image
        in_src = Input(shape=self.grayscale_shape)
        # connect the source image to the generator input
        gen_out = self.generator(in_src)
        # reforming the 3-channel Lab image
        gen_out_Lab = tf.concat([in_src, gen_out], axis=-1)
        # connect the source input and generator output to the discriminator input
        dis_out = self.discriminator([in_src, gen_out_Lab])
        # src image as input, generated image and classification output
        model = Model(in_src, [dis_out, gen_out_Lab])
        # compile model
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        return model


    def fit(self):
        raise NotImplementedError


if __name__ == "__main__":
    GAN()