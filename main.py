'''
DCGAN - MNIST Grayscale Handwritten Digits.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

from data import loadDataset
from generator import createGenerator
from discriminator import createDiscriminator
from gan import Gan

if __name__ == '__main__':
    latentDim = 100
    dataset = loadDataset()
    discriminator = createDiscriminator()
    generator = createGenerator(latentDim)
    gan = Gan(discriminator, generator, dataset)
    gan.train(latentDim)