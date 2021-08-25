from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def createDiscriminator(input=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), (2,2), padding='same', input_shape=input)) # --> 14x14
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), (2,2), padding='same')) # --> 7x7
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    discriminator = createDiscriminator()
    discriminator.summary()