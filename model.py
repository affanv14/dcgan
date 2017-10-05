from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


RANDOM_SIZE = 100

def transposed_conv(model, out_channels):
    model.add(Conv2DTranspose(out_channels, [5, 5], strides=(
        2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    return model


def conv(model, out_channels):
    model.add(Conv2D(out_channels, (5, 5),
                     kernel_initializer=RandomNormal(stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    return model


class Gan():
    def __init__(self, img_dims):
        self.random_array_size = RANDOM_SIZE
        self.imwidth = img_dims[0]
        self.imheight = img_dims[1]
        self.imchannels = img_dims[2]
        if(self.imchannels == 3):
            self.downsize_factor = 3
        else:
            self.downsize_factor = 2

    def generator(self):
        downsize = self.downsize_factor
        scale = 2**downsize
        model = Sequential()
        model.add(Dense(self.imwidth // scale * self.imheight // scale * 1024,
                        input_dim=self.random_array_size, kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(
            Reshape([self.imheight // scale, self.imwidth // scale, 1024]))
        model = transposed_conv(model, 64)
        if(downsize == 3):
            model = transposed_conv(model, 32)
        model.add(Conv2DTranspose(self.imchannels, [5, 5], strides=(
            2, 2), activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
        return model

    def discriminator(self):
        downsize = self.downsize_factor
        model = Sequential()
        model.add(Conv2D(64, (5, 5),   input_shape=(
            self.imheight, self.imwidth, self.imchannels), kernel_initializer=RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model = conv(model, 128)
        model.add(AveragePooling2D(pool_size=(2, 2)))
        if(downsize == 3):
            model = conv(model, 128)
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
