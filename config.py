configuration = {
	'dataset': '/content/drive/My Drive/Colab_thesis/Flicker8k_Dataset_new/', # Make sure you put that last slash(/)
    'features': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/modelsAndData/', # Make sure you put that last slash(/)
	'tokenFilePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr8k.token.txt',
	'trainImagePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr_8k.trainImages.txt',
	'validationImagePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr_8k.devImages.txt',
	'CNNmodelType': 'xception', # vgg16/inceptionv3/rasnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 12
}
