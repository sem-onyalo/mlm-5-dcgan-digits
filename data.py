from keras.backend import expand_dims
from keras.datasets.mnist import load_data
from keras.models import Sequential
from numpy.random import randint
from numpy.random import randn
from numpy import zeros
from numpy import ones

def loadDataset():
    # load mnist dataset
    (trainX, trainY), (_, _) = load_data()
    # expand to 3d, i.e. add channel dimension
    X = expand_dims(trainX, axis=-1)
    # filter to single digit (just for testing purposes)
    filtered = trainY == 8
    X = X[filtered]
    # convert from unsigned ints to floats
    X = X.numpy().astype('float32')
    # convert scale from 0,255 to -1,1
    X = (X - 127.5) / 127.5
    return X

def generateRealTrainingSamples(dataset, sampleNum):
    ix = randint(0, dataset.shape[0], sampleNum)
    X = dataset[ix]
    y = ones((sampleNum, 1))
    return X, y

def generateFakeTrainingSamples(generator: Sequential, latentDim, sampleNum):
    xInput = generateLatentPoints(latentDim, sampleNum)
    X = generator.predict(xInput)
    y = zeros((sampleNum, 1))
    return X, y

def generateFakeTrainingGanSamples(latentDim, sampleNum):
    X = generateLatentPoints(latentDim, sampleNum)
    y = ones((sampleNum, 1))
    return X, y

def generateLatentPoints(latentDim, sampleNum):
    xInput = randn(latentDim * sampleNum)
    xInput = xInput.reshape((sampleNum, latentDim))
    return xInput