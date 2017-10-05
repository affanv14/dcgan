from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.optimizers import Adam
from model import Gan
from PIL import Image
import numpy as np
import pickle
import argparse
import math


RANDOM_SIZE = 100

def save_imagegrid(imagearray, img_name, batch_size):
    width = imagearray.shape[2]
    height = imagearray.shape[1]
    if (imagearray.shape[3] == 1):
        mode = 'L'
        imagearray = imagearray[:, :, :, 0]
    else:
        mode = 'RGB'
    num_elements = int(math.sqrt(batch_size))
    imagegrid = Image.new(mode, (width * num_elements, height * num_elements))
    for j in range(num_elements * num_elements):
        randimg = imagearray[j] * 127.5 + 127.5
        img = Image.fromarray(randimg.astype('uint8'), mode=mode)
        imagegrid.paste(im=img, box=((j % num_elements) *
                                     width, height * (j // num_elements)))
    imagegrid.save(str(img_name) + '.png')


def train(args):
    if(args.data == 'mnist'):
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        xtrain = xtrain[:, :, :, None]
    elif(args.data == 'cifar'):
        (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

    xtrain = xtrain.astype(np.float32)
    xtrain = (xtrain - 127.5) / 127.5
    Ganmodel = Gan(img_dims=xtrain.shape[1:])
    generator = Ganmodel.generator()
    gen_sgd = Adam(lr=0.0002, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer='sgd')
    discriminator = Ganmodel.discriminator()
    discriminator_sgd = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=discriminator_sgd)
    gan = Sequential()
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=gen_sgd)
    discriminator.trainable = True
    losses = {"generator": [], "discriminator": []}
    for epoch in range(args.num_epochs):
        print("epoch number {}".format(epoch))
        for i in range((2 * (xtrain.shape[0] // args.batch_size))):
            random_array = np.random.uniform(-1,
                                             1, (args.batch_size, RANDOM_SIZE))
            generated_images = generator.predict_on_batch(random_array)

            if((i % 100) == 0):
                save_imagegrid(generated_images, 'e{}b{}'.format(
                    epoch, i), args.batch_size)

            if(i % 2 == 0):
                images = xtrain[(i // 2 * args.batch_size)
                                 :(((i // 2) + 1) * args.batch_size), :, :, :]
                labels = [1] * (args.batch_size)
            else:
                images = generated_images
                labels = [0] * (args.batch_size)

            discriminator_loss = discriminator.train_on_batch(images, labels)
            losses["discriminator"].append(discriminator_loss)
            random_array = np.random.uniform(-1,
                                             1, (args.batch_size, RANDOM_SIZE))
            labels = [1] * (args.batch_size)
            discriminator.trainable = False
            generator_loss = gan.train_on_batch(random_array, labels)
            discriminator.trainable = True
            losses["generator"].append(generator_loss)
            if args.verbose:
                print("e{}b{} discriminator loss: {}   generator loss:{}".format(epoch,
                                                                                 i,
                                                                                 discriminator_loss,
                                                                                 generator_loss))
        if ((epoch % 4) == 3):
            with open('losses.pkl', 'wb') as lossfile:
                pickle.dump(losses, lossfile)
            discriminator.save_weights('discriminator.h5')
            generator.save_weights('generator.h5')


def generate(args):
    if(args.data == 'mnist'):
        dims = (28, 28, 1)
    elif(args.data == 'cifar'):
        dims = (32, 32, 3)
    gan = Gan(img_dims=dims)
    generator = gan.generator()
    generator.load_weights('generator.h5')
    generator.compile(loss='binary_crossentropy', optimizer='sgd')
    random_array = np.random.uniform(-1,
                                     1, (args.batch_size, RANDOM_SIZE))
    generated_images = generator.predict_on_batch(random_array)
    save_imagegrid(generated_images, 'generated_images',
                   batch_size=args.batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', default='mnist',
                        help='data to use for gan(either cifar or mnist)')
    parser.add_argument('--generate', dest='generate', action='store_true',
                        help='generate images')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='number of epochs to run model for', default=20)
    parser.add_argument('--verbose', dest='verbose', action='store_false')
    parser.add_argument('--batchsize', dest='batch_size', type=int,
                        help='size of a batch', default=128)
    args = parser.parse_args()
    if args.generate:
        generate(args)
    else:
        train(args)


main()
