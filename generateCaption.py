from utils import Utils
from keras.models import Model
from config import configuration
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array

class GenerateCaption(object):
    def __init__(self, utils):
        self.utils = utils

    """ Extracts the features of test image """
    def extractImgFeature(self, filename, modelType):
        if modelType == 'inceptionv3':
            from keras.applications.inception_v3 import preprocess_input
            target_size = (299, 299)
            model = InceptionV3()
        # elif modelType == 'xception':
        #     from keras.applications.xception import preprocess_input
        #     target_size = (299, 299)
        #     model = Xception()
        # elif modelType == 'vgg16':
        #     from keras.applications.vgg16 import preprocess_input
        #     target_size = (224, 224)
        #     model = VGG16()
        # elif modelType == 'rasnet50':
        #     from keras.applications.resnet50 import preprocess_input
        #     target_size = (224, 224)
        #     model = ResNet50()

        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        image = load_img(filename, target_size=target_size) # Loading and resizing image
        image = img_to_array(image) # Convert the image pixels to a numpy array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # Reshape data for the model
        image = preprocess_input(image) # Prepare the image for the CNN Model model
        features = model.predict(image, verbose=0) # Pass image into model to get encoded features
        return features

    def start(self):
        filename = "bikestunt.jpg"
        imagefeature = self.extractImgFeature(filename, configuration['CNNmodelType'])
        captionModel = load_model(configuration['loadModelPath'])
        caption = self.utils.beamSearchCaptionGenerator(captionModel, imagefeature)
        print(caption)

if __name__ == '__main__':
    utils = Utils()
    generateCaption = GenerateCaption(utils)
    generateCaption.start()
