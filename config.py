configuration = {
	'dataset': '/content/drive/My Drive/Colab_thesis/Flicker8k_Dataset_new/', # Make sure you put that last slash(/)
    # 'featuresPath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/features/', # Make sure you put that last slash(/)
    'featuresPath': 'features/', # Make sure you put that last slash(/)
	# 'modelsPath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/models/', # Make sure you put that last slash(/)
	'modelsPath': 'models/', # Make sure you put that last slash(/)
	'loadModelPath': 'models/model_inceptionv3_GRU_epoch-08_train_loss-2.6988_val_loss-2.9413.hdf5',
	'tokenFilePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr8k.token.txt',
	'trainImagePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr_8k.trainImages.txt',
	'validationImagePath': '/content/drive/My Drive/Colab_thesis/imageCaptionBot/data/Flickr_8k.devImages.txt',
	'CNNmodelType': 'xception', # vgg16/inceptionv3/rasnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 12,
	'maxLength': 40, #This is set manually after training of model and required for test.py
	'beamIndex': 3
}
