import os
import string
import random
import numpy as np
from tqdm import tqdm
from pickle import load, dump
from config import configuration
from config import rnnConfig
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
# from tensorflow.keras.layers.merge import add


class Utils(object):
    """ This function extract features from all images and stores in a file """
    def featuresExtraction(self, dataset, modelType):
        print('Generating image features using '+str(configuration['CNNmodelType'])+' model...')
        if modelType == 'inceptionv3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            target_size = (299, 299)
            model = InceptionV3()
        elif modelType == 'xception':
            from tensorflow.keras.applications.xception import preprocess_input
            target_size = (299, 299)
            model = Xception()
        elif modelType == 'vgg16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
            target_size = (224, 224)
            model = VGG16()
        elif modelType == 'rasnet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input
            target_size = (224, 224)
            model = ResNet50()
        else:
            print("please select a appropriate model")

        model.layers.pop() # Removing the last layer from the loaded model because we dont have to classify the images
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        features = dict()
        for name in tqdm(os.listdir(dataset)):
            filename = dataset + name
            image = load_img(filename, target_size=target_size) # Loading and resizing image
            image = img_to_array(image) # Convert the image pixels to a numpy array
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # Reshape the indexs of data for the model
            image = preprocess_input(image) # Preprocess the image inorder to pass the image to the CNN Model model
            feature = model.predict(image, verbose=0) # Pass image into model to get features
            imageId = name.split('.')[0] # Store the features
            features[imageId] = feature

        dump(features, open(configuration['featuresPath']+'features_'+str(configuration['CNNmodelType'])+'.pkl', 'wb'))
        print("Size of feature vector: ", len(features))

    """ This function generates the new caption file after processing old one
    captions.txt sample data - Example :
    1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
	1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
	1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
	1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
	1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
    """
    def captionLoader(self, captionFile, processedCaptionFile):
        file = open(captionFile, 'r')
        doc = file.read()
        file.close()
        """
    	Captions dict is of form:
    	{
    		image_id1 : [caption1, caption2, etc],
    		image_id2 : [caption1, caption2, etc],
    		...
    	}
    	"""
        mapping = dict()
        for line in doc.split('\n'): # Split by white space
            tokens = line.split()
            if len(line) < 2:
                continue

            imageId, image_caption = tokens[0], tokens[1:] # Name of the image is id and rest is caption.
            imageId = imageId.split('.')[0]
            image_caption = ' '.join(image_caption) # Caption tokens are converted to string
            if imageId not in mapping:
                mapping[imageId] = list()
            mapping[imageId].append(image_caption)

        # # Cleaning the captions
        # table = str.maketrans('', '', string.punctuation)
        # for _, caption_list in mapping.items():
        #     for i in range(len(caption_list)):
        #         caption = caption_list[i]
        #         caption = caption.split()
        #         caption = [word.lower() for word in caption]
        #         # Remove punctuation from each token
        #         caption = [w.translate(table) for w in caption]
        #         # Remove hanging 's' and 'a'
        #         caption = [word for word in caption if len(word)>1]
        #         # Remove tokens with numbers in them
        #         caption = [word for word in caption if word.isalpha()]
        #         caption_list[i] = ' '.join(caption)

        # creating a processed caption file
        lines = list()
        for key, mapping_list in mapping.items():
            for caption in mapping_list:
                lines.append(key + ' ' + caption)
        data = '\n'.join(lines)
        file = open(processedCaptionFile, 'w')
        file.write(data)
        file.close()
        print("Processed caption file generated: captions.txt")

    """ This function is used to load the Images from the txt file"""
    def loadImageIdSet(self, filename):
        file = open(filename, 'r')
        doc = file.read()
        file.close()
        ids = list()
        for line in doc.split('\n'):
            if len(line) < 1:
                continue
            _id = line.split('.')[0] # Image Identifier
            ids.append(_id)
        return set(ids)

    """ The model we'll develop will generate a caption for a given image and the caption will be generated one word at a time.
    The sequence of previously generated words will be provided as input. Therefore, we will need a ‘first word’ to
	kick-off the generation process and a ‘last word‘ to signal the end of the caption.
	We'll use the strings ‘startseq‘ and ‘endseq‘ for this purpose. These tokens are added to the captions as they are loaded.
	It is important to do this now before we encode the text so that the tokens are also encoded correctly.
	"""
    def cleanedCaptionsLoader(self, captionfile, imagefile):
        ids = self.loadImageIdSet(imagefile)
        file = open(captionfile, 'r')
        doc = file.read()
        file.close()
        captions = dict()
        count = 0
        for line in doc.split('\n'):
            tokens = line.split()
            imageId, image_caption = tokens[0], tokens[1:] # Spliting image id and caption
            if imageId in ids: # Skipping the image which are not in the imagefile
                if imageId not in captions:
                    captions[imageId] = list()

                caption = 'startseq ' + ' '.join(image_caption) + ' endseq' # Wrap caption in start & end tokens
                captions[imageId].append(caption)
                count = count+1
        return captions, count

    """ Load image features """
    def imageFeaturesLoader(self, featureFile, imagefile):
        ids = self.loadImageIdSet(imagefile)
        all_features = load(open(featureFile, 'rb'))
        features = {_id: all_features[_id] for _id in ids} # Only keeping the image which are in the imagefile
        return features

    """ Convert a dictionary to a list """
    def toList(self, captions):
        all_captions = list()
        for imageId in captions.keys():
            [all_captions.append(caption) for caption in captions[imageId]]
        return all_captions

    '''The captions will need to be encoded to numbers before it can be presented to the model.
    The first step in encoding the captions is to create a consistent mapping from words to unique integer values.
	Keras provides the Tokenizer class that can learn this mapping from the loaded captions.
	Fit a tokenizer on given captions.
    '''
    def createTokenizer(self, trainCaptions):
        lines = self.toList(trainCaptions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines) # ex: {'the':1, 'a':2, ...}
        dump(tokenizer, open(configuration['featuresPath']+'tokenizer.pkl', 'wb'))

    """ Calculate the max length of the captions from the given captions """
    def maxLengthOfCaption(self, captions):
        lines = self.toList(captions)
        return max(len(line.split()) for line in lines)

    """Each caption will be split into words. The model will be provided one word & the image and it generates the next word.
	Then the first two words of the caption will be provided to the model as input with the image to generate the next word.
	This is how the model will be trained.
	For example, the input sequence “little girl running in field” would be
		split into 6 input-output pairs to train the model:

		iFeature    tFeature(text sequence) 			     outWord(word)
		-------------------------------------------------------------------
		image	    startseq,										little
		image	    startseq, little,								girl
		image	    startseq, little, girl,							running
		image	    startseq, little, girl, running,				in
		image	    startseq, little, girl, running, in,			field
		image	    startseq, little, girl, running, in, field,		endseq
    """
    # Create sequences of images, input sequences and output words for an image
    def createSequences(self, tokenizer, maxCaption, captionsList, image):
        iFeature, tFeature, outWord = list(), list(), list() #input for image feature, text feature and output word
        vocabSize = len(tokenizer.word_index) + 1
        for caption in captionsList:
            # each caption is converted to numbers according to the toenizer.
            # Ex: [3, 1, 19, 316, 65, 1, 197, 120, 2]
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                inSeq, outSeq = seq[:i], seq[i] # Input and output sequence
                # pad_sequences ex: sequence = [[1], [2, 3], [4, 5, 6]] if max len is 2 then the output will be:
                # [[0,1],[2,3],[5,6]]
                inSeqRes = pad_sequences([inSeq], maxlen=maxCaption)[0] # changing the lenth of list to same as maxCaption size
                outSeqRes = to_categorical([outSeq], num_classes=vocabSize)[0] # Transform the number to vector
                iFeature.append(image)
                tFeature.append(inSeqRes)
                outWord.append(outSeqRes)
        return iFeature, tFeature, outWord

    """ Data generator, intended to be used in a call to model.fit() """
    def dataGenerator(self, imagefeatures, captions, tokenizer, maxCaption, batchSize):
        imageIds = list(captions.keys()) # Image ids
        count=0
        while True:
            if count >= len(imageIds):
                count = 0 # Generator exceeded or reached the end so restart it
            inputImg_batch, inputSequence_batch, outputWord_batch = list(), list(), list() # Batch list to store data
            for i in range(count, min(len(imageIds), count+batchSize)):
                imageId = imageIds[i] # Image id
                image = imagefeatures[imageId][0] # Image feature
                captionsList = captions[imageId] # Image caption
                random.shuffle(captionsList)
                inputImg, inputSequence, outputWord = self.createSequences(tokenizer, maxCaption, captionsList, image)

    			# Add to batch
                for j in range(len(inputImg)):
                    inputImg_batch.append(inputImg[j])
                    inputSequence_batch.append(inputSequence[j])
                    outputWord_batch.append(outputWord[j])
            count = count + batchSize
            yield [np.array(inputImg_batch), np.array(inputSequence_batch)], np.array(outputWord_batch)

    """ The RNN model """
    def captionModel(self, vocabSize, maxCaption, modelType, RNNmodel):
        shape = 1000
        # squeezing features from the CNN model
        imageInput = Input(shape=(shape,))
        imageModel_1 = Dropout(rnnConfig['dropout'])(imageInput)
        imageModel = Dense(rnnConfig['embedding_size'], activation='relu')(imageModel_1)

        # Sequence Model
        captionInput = Input(shape=(maxCaption,))
        captionModel_1 = Embedding(vocabSize, rnnConfig['embedding_size'], mask_zero=True)(captionInput)
        captionModel_2 = Dropout(rnnConfig['dropout'])(captionModel_1)
        if RNNmodel == 'LSTM':
            captionModel = LSTM(rnnConfig['LSTM_GRU_units'])(captionModel_2)
        elif RNNmodel == 'GRU':
            captionModel = GRU(rnnConfig['LSTM_GRU_units'])(captionModel_2)

        # Merging the models and creating a softmax classifier
        finalModel_1 = concatenate([imageModel, captionModel])
        # finalModel_1 = add([imageModel, captionModel])
        finalModel_2 = Dense(rnnConfig['dense_units'], activation='relu')(finalModel_1)
        finalModel = Dense(vocabSize, activation='softmax')(finalModel_2)

        # tieing it together
        model = Model(inputs=[imageInput, captionInput], outputs=finalModel)
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])
        return model

    """ Alternate RNN model """
    def updatedCaptionModel(self, vocabSize, maxCaption, modelType, RNNmodel):
        shape = 1000
        # squeezing features from the CNN model
        imageInput = Input(shape=(shape,))
        imageModel_1 = Dense(rnnConfig['embedding_size'], activation='relu')(imageInput)
        imageModel = RepeatVector(maxCaption)(imageModel_1)

        # Sequence Model
        captionInput = Input(shape=(maxCaption,))
        captionModel_1 = Embedding(vocabSize, rnnConfig['embedding_size'], mask_zero=True)(captionInput)
        if RNNmodel == 'LSTM':
            captionModel_2 = LSTM(rnnConfig['LSTM_GRU_units'], return_sequences=True)(captionModel_1)
        elif RNNmodel == 'GRU':
            captionModel_2 = GRU(rnnConfig['LSTM_GRU_units'], return_sequences=True)(captionModel_1)
        captionModel = TimeDistributed(Dense(rnnConfig['embedding_size']))(captionModel_2)

        # Merging the models and creating a softmax classifier
        finalModel_1 = concatenate([imageModel, captionModel])
        finalModel_2 = Bidirectional(GRU(rnnConfig['LSTM_GRU_units'], return_sequences=False))(finalModel_1)
        finalModel = Dense(vocabSize, activation='softmax')(finalModel_2)

        # tieing it together
        model = Model(inputs=[imageInput, captionInput], outputs=finalModel)
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])
        return model

    """ Map an integer to a word """
    def intToWord(self, integer, tokenizer):
    	for word, index in tokenizer.word_index.items():
    		if index == integer:
    			return word
    	return None

    """Generate a caption for an image, given a pre-trained model and a tokenizer to map integer back to word
    Using BEAM Search algorithm
    """
    def beamSearchCaptionGenerator(self, model, image, tokenizer):
        inText = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
        while len(inText[0][0]) < configuration['maxLength']:
            tempList = []
            for seq in inText:
                paddedSeq = pad_sequences([seq[0]], maxlen=configuration['maxLength']) # changing the lenth of list to same as maxCaption size
                preds = model.predict([image,paddedSeq], verbose=0)
    			# Take top `beam_index` predictions (i.e. which have highest probailities)
                top_preds = np.argsort(preds[0])[-configuration['beamIndex']:]
                for word in top_preds:
                    next_seq, prob = seq[0][:], seq[1]
                    next_seq.append(word)
                    prob += preds[0][word] # Update probability
                    tempList.append([next_seq, prob]) # Append as input for generating the next word

            inText = tempList
            inText = sorted(inText, reverse=False, key=lambda l: l[1]) # Sorting according to the probabilities
            inText = inText[-configuration['beamIndex']:] # Take the top words
        inText = inText[-1][0] # caption in number form
        finalCaption_raw = [self.intToWord(i,tokenizer) for i in inText]
        finalCaption = []
        for word in finalCaption_raw:
            if word=='endseq':
                break
            else:
                finalCaption.append(word)
        finalCaption.append('endseq')
        return ' '.join(finalCaption)
