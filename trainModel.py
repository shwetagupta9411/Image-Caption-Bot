import os
from utils import Utils
from models import Models
from pickle import load, dump
from config import configuration
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

class train(object):
    def __init__(self, utils, models):
        self.utils = utils
        self.models = models

    def dataLoader(self, loadImages):
        dataCaptions, dataCount = self.utils.cleanedCaptionsLoader(configuration['features']+'captions.txt', loadImages)
        dataFeatures = self.utils.imageFeaturesLoader(configuration['features']+'features_'+str(configuration['CNNmodelType'])+'.pkl', loadImages)
        print("Available captions : ", dataCount)
        print("Available images : ", len(dataFeatures))
        return dataCaptions, dataFeatures

    def start(self):
        """ Loading training data """
        print("\n-------------- Available training data --------------")
        trainCaptions, trainFeatures = self.dataLoader(configuration['trainImagePath'])

        """ Loading validation data """
        print("------------- Available validation data -------------")
        valCaptions, valFeatures = self.dataLoader(configuration['validationImagePath'])

        """ Generating tokens for training data and largest caption length"""
        print("\n----------------------------------------------------")
        maxCaption = self.utils.maxLengthOfCaption(trainCaptions)
        print("Length of largest caption of training: ", maxCaption)
        if not os.path.exists(configuration['features']+'tokenizer.pkl'):
            self.utils.createTokenizer(trainCaptions)
            print("Tokenizer file generated")
        else:
            print('Tokenizer file already present at %s' % (configuration['features']+'tokenizer.pkl') )
        tokenizer = load(open(configuration['features']+'tokenizer.pkl', 'rb'))
        vocabSize = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocabSize)

        print("\n----------------------------------------------------")
        stepsToTrain = round(len(trainFeatures)/configuration['batchSize'])
        stepsToVal = round(len(valFeatures)/configuration['batchSize'])
        print("Batch size: %d" % configuration['batchSize'])
        print("epochs: %d" % configuration['epochs'])
        print("Steps per epoch for training: %d" % stepsToTrain)
        print("Steps per epoch for validation: %d\n" % stepsToVal)

        model = models.captionModel(vocabSize, maxCaption, configuration['CNNmodelType'], configuration['RNNmodelType'])
        print('\nRNN Model Summary : ')
        print(model.summary())

        modelSavePath = configuration['models']+"model_"+str(configuration['CNNmodelType'])+"_"+str(configuration['RNNmodelType'])+"_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(modelSavePath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks = [checkpoint]

        if configuration['batchSize'] <= len(trainCaptions.keys()):
            trainingDataGen = self.utils.dataGenerator(trainFeatures, trainCaptions, tokenizer, maxCaption, configuration['batchSize'])
            validationDataGen = self.utils.dataGenerator(valFeatures, valCaptions, tokenizer, maxCaption, configuration['batchSize'])

            history = model.fit_generator(trainingDataGen,
                epochs=configuration['epochs'],
                steps_per_epoch=stepsToTrain,
                validation_data=validationDataGen,
                validation_steps=stepsToVal,
                callbacks=callbacks,
                verbose=1)
            # list all data in history
            print(history.history.keys())
            print(history.history['accuracy'])
            print(history.history['val_accuracy'])
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        else:
            print("Batch size must be less than or equal to " + len(imageIds))

        # [imageInputF,textSeqInput],outTestInput = next(trainingDataGen)
        # print(imageInputF.shape, textSeqInput.shape, outTestInput.shape)



if __name__ == '__main__':
    utils = Utils()
    models = Models()
    print("update check ")
    print("\t\t----------------- Using CNN model %s and RNN model %s -----------------\n" % (configuration['CNNmodelType'], configuration['RNNmodelType']))
    if os.path.exists(configuration['features']+'features_'+str(configuration['CNNmodelType'])+'.pkl'):
        print('Features are already generated at %s' % (configuration['features']+'features_'+str(configuration['CNNmodelType'])+'.pkl') )
    else:
        utils.featuresExtraction(configuration['dataset'], configuration['CNNmodelType'])
        print("features saved successfully" % ())

    if os.path.exists(configuration['features']+'captions.txt'):
        print('Captions are already generated at %s' % (configuration['features']+'captions.txt'))
    else:
        utils.captionLoader(configuration['tokenFilePath'], configuration['features']+'captions.txt')

    train = train(utils, models)
    train.start()
