from utils import Utils
from pickle import load
from tensorflow.keras.models import Model
from config import configuration, model_path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class GenerateCaption(object):
    def __init__(self, filename, modelType):
        self.filename = filename
        self.modelType = modelType

    """ Extracts the features of test image """
    def extractImgFeature(self, filename, modelType):
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
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        image = load_img(filename, target_size=target_size) # Loading and resizing image
        image = img_to_array(image) # Convert the image pixels to a numpy array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # Reshape data for the model
        image = preprocess_input(image) # Prepare the image for the CNN Model model
        features = model.predict(image, verbose=0) # Pass image into model to get encoded features
        return features

    def start(self):
        utils = Utils()
        imagefeature = self.extractImgFeature(self.filename, self.modelType)
        model_path(self.modelType)
        print(configuration['loadModelPath'])
        captionModel = load_model(configuration['loadModelPath'])
        tokenizer = load(open(configuration['featuresPath']+'tokenizer.pkl', 'rb'))
        """ Generating caption using decoder RNN model + Beam Search """
        genCaption_beam = utils.beamSearchCaptionGenerator(captionModel, imagefeature, tokenizer)
        caption_beam = 'I think its ' + genCaption_beam.split()[1]
        # caption_beam = 'I am not really confident, but I think its ' + genCaption_beam.split()[1]
        # Remove startseq and endseq
        for x in genCaption_beam.split()[2:len(genCaption_beam.split())-1]:
            caption_beam = caption_beam + ' ' + x
        caption_beam += '.'
        """ Generating caption using decoder RNN model + Greedy Search """
        genCaption_greedy = utils.greedySearchCaptionGenerator(captionModel, imagefeature, tokenizer)
        caption_greedy = 'I think its ' + genCaption_greedy.split()[1]
        # caption_greedy = 'I am not really confident, but I think its ' + genCaption_beam.split()[1]
        # Remove startseq and endseq
        for x in genCaption_greedy.split()[2:len(genCaption_greedy.split())-1]:
            caption_greedy = caption_greedy + ' ' + x
        caption_greedy += '.'

        # print("caption_greedy:", caption_greedy)
        # print("caption_beam:", caption_beam)
        return caption_beam, caption_greedy

# if __name__ == '__main__':
#     filename = "bikestunt.jpg" # pass filename
#     modelType = "inceptionv3" # pass modeltype: vgg16/inceptionv3/rasnet50/xception
#     generateCaption = GenerateCaption(filename, modelType)
#     generateCaption.start()
