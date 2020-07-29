import os
import string
import random
import numpy as np
from tqdm import tqdm
from pickle import load, dump
from keras.models import Model
from config import configuration
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing.text import Tokenizer
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array


class Utils(object):
    """ This function extract features from all images and stores in a file """
    def featuresExtraction(self, dataset, modelType):
        print('Generating image features using '+str(configuration['CNNmodelType'])+' model...')
        if modelType == 'inceptionv3':
            from keras.applications.inception_v3 import preprocess_input
            target_size = (299, 299)
            model = InceptionV3()
        elif modelType == 'xception':
            from keras.applications.xception import preprocess_input
            target_size = (299, 299)
            model = Xception()
        elif modelType == 'vgg16':
            from keras.applications.vgg16 import preprocess_input
            target_size = (224, 224)
            model = VGG16()
        elif modelType == 'rasnet50':
            from keras.applications.resnet50 import preprocess_input
            target_size = (224, 224)
            model = ResNet50()

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

        dump(features, open(configuration['features']+'features_'+str(configuration['CNNmodelType'])+'.pkl', 'wb'))
        print("Size of feature vector: ", len(features))

    """ This function generates the new caption file after processing old one
    captions.txt sample data - Example : 2252123185_487f21e336 stadium full of people watch game"""
    def captionLoader(self, captionFile, processedCaptionFile):
        file = open(captionFile, 'r')
        doc = file.read()
        file.close()
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


        lines = list()
        for key, mapping_list in mapping.items():
            for caption in mapping_list:
                lines.append(key + ' ' + caption)
        data = '\n'.join(lines)
        file = open(processedCaptionFile, 'w')
        file.write(data)
        file.close()
        print("Processed caption file generated: captions.txt")

    """ This function is used to load the file trainImages.txt """
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

                caption = 'startseq ' + ' '.join(image_caption) + ' endsseq' # Wrap caption in start & end tokens
                captions[imageId].append(caption)
                count = count+1
        return captions, count

    def imageFeaturesLoader(self, featureFile, imagefile):
        ids = self.loadImageIdSet(imagefile)
        all_features = load(open(featureFile, 'rb'))
        features = {_id: all_features[_id] for _id in ids} # Only keeping the image which are in the imagefile
        return features

    def toList(self, captions):
        all_captions = list()
        for imageId in captions.keys():
            [all_captions.append(caption) for caption in captions[imageId]]
        return all_captions

    def createTokenizer(self, trainCaptions):
        lines = self.toList(trainCaptions)
        print(len(lines))
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines) # ex: {'the':1, 'a':2, ...}
        dump(tokenizer, open(configuration['features']+'tokenizer.pkl', 'wb'))

    def maxLengthOfCaption(self, captions):
    	lines = self.toList(captions)
    	return max(len(line.split()) for line in lines)

    '''
	*Each caption will be split into words. The model will be provided one word & the image and it generates the next word.
	*Then the first two words of the caption will be provided to the model as input with the image to generate the next word.
	*This is how the model will be trained.
	*For example, the input sequence “little girl running in field” would be
		split into 6 input-output pairs to train the model:

		iFeature    tFeature(text sequence) 			     outWord(word)
		-------------------------------------------------------------------
		image	    startseq,										little
		image	    startseq, little,								girl
		image	    startseq, little, girl,							running
		image	    startseq, little, girl, running,				in
		image	    startseq, little, girl, running, in,			field
		image	    startseq, little, girl, running, in, field,		endseq
    '''
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

    def dataGenerator(self, imagefeatures, captions, tokenizer, maxCaption, batchSize):
        # random.seed(1035)
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
            yield [[np.array(inputImg_batch), np.array(inputSequence_batch)], np.array(outputWord_batch)]
