from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, GRU, Embedding, concatenate
from keras.layers.merge import add
# RepeatVector, TimeDistributed, Bidirectional
class Models(object):
    def captionModel(self, vocabSize, maxCaption, modelType, RNNmodel):
        if modelType == 'inceptionv3' or modelType == 'xception':
            shape = 2048 # InceptionV3 and xception outputs a 2048 dimensional vector for each image
        elif modelType == 'vgg16' or modelType == 'rasnet50':
            shape = 4096 # VGG16 and rasnet50 outputs a 4096 dimensional vector for each image

        # squeezing features from the CNN model
        imageInput = Input(shape=(shape,))
        imageModel_1 = Dropout(0.5)(imageInput)
        imageModel = Dense(300, activation='relu')(imageModel_1)

        # Sequence Model
        captionInput = Input(shape=(maxCaption,))
        captionModel_1 = Embedding(vocabSize, 300, mask_zero=True)(captionInput)
        captionModel_2 = Dropout(0.5)(captionModel_1)
        if RNNmodel == 'LSTM':
            captionModel = LSTM(256)(captionModel_2)
        elif RNNmodel == 'GRU':
            captionModel = GRU(256)(captionModel_2)

        # Merging the models and creating a softmax classifier
        finalModel_1 = concatenate([imageModel, captionModel])
        # finalModel_1 = add([imageModel, captionModel])
        finalModel_2 = Dense(256, activation='relu')(finalModel_1)
        finalModel = Dense(vocabSize, activation='softmax')(finalModel_2)

        # tieing it together
        model = Model(inputs=[imageInput, captionInput], outputs=finalModel)
        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model
