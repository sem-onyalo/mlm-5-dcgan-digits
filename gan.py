import os
from generator import createGenerator
from discriminator import createDiscriminator
from data import generateRealTrainingSamples, generateFakeTrainingSamples, generateFakeTrainingGanSamples
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Gan:
    def __init__(self, discriminator: Sequential, generator: Sequential, dataset):
        self.discriminator = discriminator
        self.generator = generator
        self.dataset = dataset
        self.model = self.createModel()

        self.dLossRealHistory = list()
        self.dAccRealHistory = list()
        self.dLossFakeHistory = list()
        self.dAccFakeHistory = list()
        self.gLossHistory = list()

    def createModel(self):
        self.discriminator.trainable = False

        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, latentDim, epochs=100, batches=256, evalFreq=10):
        if not os.path.exists('eval'):
            os.makedirs('eval')

        batchesPerEpoch = int(self.dataset.shape[0] / epochs) # batches
        halfBatch = int(batches / 2)

        self.plotStartingImageSamples(latentDim)
        
        for i in range(epochs):
            for j in range(batchesPerEpoch):
                xReal, yReal = generateRealTrainingSamples(self.dataset, halfBatch)
                dLossReal, dAccReal = self.discriminator.train_on_batch(xReal, yReal)

                xFake, yFake = generateFakeTrainingSamples(self.generator, latentDim, halfBatch)
                dLossFake, dAccFake = self.discriminator.train_on_batch(xFake, yFake)

                xGan, yGan = generateFakeTrainingGanSamples(latentDim, batches)
                gLoss = self.model.train_on_batch(xGan, yGan)

                self.dLossRealHistory.append(dLossReal)
                self.dAccRealHistory.append(dAccReal)
                self.dLossFakeHistory.append(dLossFake)
                self.dAccFakeHistory.append(dAccFake)
                self.gLossHistory.append(gLoss)

                print('>%d, %d/%d, dr=%.3f, df=%.3f, g=%.3f' % 
                    (i + 1, j + 1, batchesPerEpoch, dLossReal, dLossFake, gLoss))
            
            if (i + 1) % evalFreq == 0:
                self.evaluate(i, latentDim)

        self.plotHistory()

    def evaluate(self, epoch, latentDim, samples=150):
        xReal, yReal = generateRealTrainingSamples(self.dataset, samples)
        _, accReal = self.discriminator.evaluate(xReal, yReal)

        xFake, yFake = generateFakeTrainingSamples(self.generator, latentDim, samples)
        _, accFake = self.discriminator.evaluate(xFake, yFake)

        print(epoch, accReal, accFake)
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accReal * 100, accFake * 100))

        filename = 'eval/generated_model_e%03d.h5' % (epoch + 1)
        self.generator.save(filename)

        self.plotImageSamples(xFake, epoch)

    def plotImageSamples(self, samples, epoch, n=10):
        # scale from -1,1 to 0,1
        scaledSamples = (samples + 1) / 2.0

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledSamples[i, :, :, 0], cmap='gray_r')

        filename = 'eval/generated_plot_e%03d.png' % (epoch + 1)
        pyplot.savefig(filename)
        pyplot.close()

    def plotStartingImageSamples(self, latentDim, samples=150):
        xFake, _ = generateFakeTrainingSamples(self.generator, latentDim, samples)
        self.plotImageSamples(xFake, -1)

    def plotHistory(self):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.dLossRealHistory, label='dLossReal')
        pyplot.plot(self.dLossFakeHistory, label='dLossFake')
        pyplot.plot(self.gLossHistory, label='gLoss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.dAccRealHistory, label='accReal')
        pyplot.plot(self.dAccFakeHistory, label='accFake')
        pyplot.legend()

        pyplot.savefig('eval/loss_acc_history.png')
        pyplot.close()

if __name__ == '__main__':
    discriminator = createDiscriminator()
    generator = createGenerator()
    gan = Gan(discriminator, generator, None)
    gan.model.summary()