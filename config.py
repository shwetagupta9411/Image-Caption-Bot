configuration = {
	'dataset': '/content/drive/My Drive/Colab_thesis/Flicker8k_Dataset_new/', # Make sure you put that last slash(/)
    'featuresPath': 'features/', # Make sure you put that last slash(/)
	'modelsPath': 'models/', # Make sure you put that last slash(/)
	'loadModelPath': '',
	'tokenFilePath': 'data/Flickr8k.token.txt',
	'trainImagePath': 'data/Flickr_8k.trainImages.txt',
	'validationImagePath': 'data/Flickr_8k.devImages.txt',
	'CNNmodelType': 'inceptionv3', # vgg16/inceptionv3/rasnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 12,
	'maxLength': 40, # This is set manually after training of model and required for test.py
	'beamIndex': 3
}
if configuration['CNNmodelType'] == 'inceptionv3':
	configuration['loadModelPath'] = 'models/model_inceptionv3_GRU_epoch-11_train_loss-2.5534_val_loss-2.9011.hdf5'
elif configuration['CNNmodelType'] == 'vgg16':
	configuration['loadModelPath'] = 'models/model_vgg16_GRU_epoch-08_train_loss-2.5262_val_loss-3.0908.hdf5'
elif configuration['CNNmodelType'] == 'xception':
	configuration['loadModelPath'] = 'models/model_xception_GRU_epoch-12_train_loss-2.4102_val_loss-2.8929.hdf5'
elif configuration['CNNmodelType'] == 'rasnet50':
	configuration['loadModelPath'] = 'models/model_rasnet50_GRU_epoch-09_train_loss-2.5509_val_loss-2.9663.hdf5'
