from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, Reshape
from tensorflow.keras.models import Sequential

def createGenerator(input=100):
    inputDim = 7
    inputFilters = 128
    convFilters = 128
    inputNodes = inputFilters * inputDim * inputDim
    model = Sequential()
    model.add(Dense(inputNodes, input_dim=input))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((inputDim, inputDim, inputFilters)))
    model.add(Conv2DTranspose(convFilters, (4,4), (2, 2), padding='same')) # --> 14x14
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(convFilters, (4,4), (2,2), padding='same')) # --> 28x28
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model

if __name__ == '__main__':
    generator = createGenerator()
    generator.summary()