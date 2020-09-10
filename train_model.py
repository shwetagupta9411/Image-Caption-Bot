import os
from tqdm import tqdm
from utils import Utils
from datetime import datetime
from pickle import load, dump
import matplotlib.pyplot as plt
from config import configuration, model_path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint



class Train(object):
    def __init__(self, utils):
        self.utils = utils

    def start(self):
        """ Loading training data """
        print("\n-------------- Available training data --------------")
        trainCaptions, trainFeatures = self.utils.dataLoader(configuration['trainImagePath'])

        """ Loading validation data """
        print("------------- Available validation data -------------")
        valCaptions, valFeatures = self.utils.dataLoader(configuration['validationImagePath'])

        """ Generating tokens for training data and largest caption length"""
        print("\n----------------------------------------------------")
        maxCaption = self.utils.maxLengthOfCaption(trainCaptions)
        print("Length of largest caption of training: ", maxCaption)
        if not os.path.exists(configuration['featuresPath']+'tokenizer.pkl'):
            self.utils.createTokenizer(trainCaptions)
            print("Tokenizer file generated")
        else:
            print('Tokenizer file already present at %s' % (configuration['featuresPath']+'tokenizer.pkl') )
        tokenizer = load(open(configuration['featuresPath']+'tokenizer.pkl', 'rb'))
        vocabSize = len(tokenizer.word_index) + 1
        print('Training vocabulary Size: %d' % vocabSize)

        print("\n----------------------------------------------------")
        stepsToTrain = round(len(trainFeatures)/configuration['batchSize'])
        stepsToVal = round(len(valFeatures)/configuration['batchSize'])
        print("Batch size: %d" % configuration['batchSize'])
        print("epochs: %d" % configuration['epochs'])
        print("Steps per epoch for training: %d" % stepsToTrain)
        print("Steps per epoch for validation: %d\n" % stepsToVal)

        model = self.utils.captionModel(vocabSize, maxCaption, configuration['CNNmodelType'], configuration['RNNmodelType'])
        print(model.summary())

        "To plot a model Image"
        # plot_model(model, "caption_model.png", show_shapes=True) # works on colab only
        "The modified caption model"
        # model = self.utils.updatedCaptionModel(vocabSize, maxCaption, configuration['CNNmodelType'], configuration['RNNmodelType'])
        # print('\nRNN Model Summary : ')


        modelSavePath = configuration['modelsPath']+"model_"+str(configuration['CNNmodelType'])+"_"+str(configuration['RNNmodelType'])+"_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(modelSavePath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboardCallback = TensorBoard(log_dir=logdir, histogram_freq=1)
        # callbacks = [checkpoint, tensorboardCallback]
        callbacks = [checkpoint]

        if configuration['batchSize'] <= len(trainCaptions.keys()):
            trainingDataGen = self.utils.dataGenerator(trainFeatures, trainCaptions, tokenizer, maxCaption, configuration['batchSize'])
            validationDataGen = self.utils.dataGenerator(valFeatures, valCaptions, tokenizer, maxCaption, configuration['batchSize'])

            """ Use to train the saved model """
            # model_path(configuration['CNNmodelType'])
            # print(configuration['loadModelPath'])
            # model = load_model(configuration['loadModelPath'])

            """ Training the model """
            history = model.fit(trainingDataGen,
                epochs=configuration['epochs'],
                steps_per_epoch=stepsToTrain,
                validation_data=validationDataGen,
                validation_steps=stepsToVal,
                callbacks=callbacks,
                verbose=1)
            print("Model trained successfully.")

            # list all data in history
            print(history.history)
            fName = configuration['modelsPath'] + "history/" + configuration['CNNmodelType'] + "_" + configuration['RNNmodelType'] + "_" + 'model_history.txt'
            file = open(fName, 'w')
            file.write(str(history.history)+'\n')
            file.close()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

        else:
            print("Batch size must be less than or equal to " + list(trainCaptions.keys()))


if __name__ == '__main__':
    utils = Utils()
    print("\t\t----------------- Using CNN model %s and RNN model %s -----------------\n" % (configuration['CNNmodelType'], configuration['RNNmodelType']))
    if os.path.exists(configuration['featuresPath']+'features_'+str(configuration['CNNmodelType'])+'.pkl'):
        print('Features are already generated at %s' % (configuration['featuresPath']+'features_'+str(configuration['CNNmodelType'])+'.pkl') )
    else:
        utils.featuresExtraction(configuration['dataset'], configuration['CNNmodelType'])
        print("features saved successfully" % ())

    if os.path.exists(configuration['featuresPath']+'captions.txt'):
        print('Processed caption file is already generated at %s' % (configuration['featuresPath']+'captions.txt'))
    else:
        utils.captionLoader(configuration['tokenFilePath'], configuration['featuresPath']+'captions.txt')

    train = Train(utils)
    train.start()
