configuration = {
	'dataset': '/content/drive/My Drive/Colab_thesis/Flicker8k_Dataset_new/', # Make sure you put that last slash(/)
    'featuresPath': 'features/', # Make sure you put that last slash(/)
	'modelsPath': 'models/', # Make sure you put that last slash(/)
	'loadModelPath': '',
	'tokenFilePath': 'data/Flickr8k.token.txt',
	'trainImagePath': 'data/Flickr_8k.trainImages.txt',
	'validationImagePath': 'data/Flickr_8k.devImages.txt',
	'CNNmodelType': 'rasnet50', # vgg16/inceptionv3/rasnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 2,
	'maxLength': 40, # This is set manually after training of model and required for test.py
	'beamIndex': 3
}
def model_path(modelType):
	if modelType == 'inceptionv3':
		configuration['loadModelPath'] = 'models/model_inceptionv3_GRU_epoch-16_train_loss-2.7460_val_loss-3.2425.hdf5'
	elif modelType == 'vgg16':
		configuration['loadModelPath'] = 'models/model_inceptionv3_GRU_epoch-16_train_loss-2.7460_val_loss-3.2425.hdf5'
	elif modelType == 'xception':
		configuration['loadModelPath'] = 'models/model_xception_GRU_epoch-15_train_loss-2.7480_val_loss-3.2163.hdf5'
	elif modelType == 'rasnet50':
		configuration['loadModelPath'] = 'models/model_rasnet50_GRU_epoch-14_train_loss-2.8236_val_loss-3.2510.hdf5'

models_summary = {
	"Xception":{
		"val_loss": [4.9473, 4.3192, 3.9664, 3.7579, 3.6171, 3.5152, 3.4394, 3.3801, 3.3352, 3.2943, 3.2667, 3.2495, 3.2314, 3.2204, 3.2163, 3.2196, 3.2250, 3.2320, 3.2428],
		"train_loss": [5.9991, 4.7110, 4.1710, 3.8689, 3.6611, 3.4976, 3.3662, 3.2565, 3.1610, 3.0772, 3.0013, 2.9299, 2.8655, 2.8039, 2.7480, 2.6951, 2.6437, 2.5963, 2.5500],
		"epoch": 15,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.606086, 0.606086, 0.606086, 0.606086], #yet to add
		"model_val_loss":3.2163,
		"model_train_loss":2.7480
	},
	"InceptionV3":{
		"val_loss": [5.009276866912842, 4.370221138000488, 4.016548156738281, 3.797884464263916, 3.6575920581817627, 3.5547163486480713, 3.4766595363616943, 3.418569326400757, 3.369173049926758, 3.3337106704711914, 3.3041746616363525, 3.2746121883392334, 3.2598, 3.2511, 3.2458, 3.2425, 3.2436, 3.2458, 3.2554, 3.2648],
		"train_loss": [6.026840686798096, 4.7727131843566895, 4.218074321746826, 3.915517807006836, 3.7000908851623535, 3.5450735092163086, 3.414228677749634, 3.304046869277954, 3.208782911300659, 3.124225378036499, 3.0475149154663086, 2.977383613586426, 2.9120, 2.8537, 2.7986, 2.7460, 2.6964, 2.6486, 2.6023, 2.5592],
		"epoch": 16,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.606086, 0.606086, 0.606086, 0.606086], #yet to do
		"model_val_loss":3.2425,
		"model_train_loss":2.7460
	},
	"VGG16":{
		"val_loss": [5.009276866912842, 4.370221138000488, 4.016548156738281, 3.797884464263916, 3.6575920581817627, 3.5547163486480713, 3.4766595363616943, 3.418569326400757, 3.369173049926758, 3.3337106704711914, 3.3041746616363525, 3.2746121883392334, 3.2598, 3.2511, 3.2458, 3.2425, 3.2436, 3.2458, 3.2554, 3.2648],
		"train_loss": [6.026840686798096, 4.7727131843566895, 4.218074321746826, 3.915517807006836, 3.7000908851623535, 3.5450735092163086, 3.414228677749634, 3.304046869277954, 3.208782911300659, 3.124225378036499, 3.0475149154663086, 2.977383613586426, 2.9120, 2.8537, 2.7986, 2.7460, 2.6964, 2.6486, 2.6023, 2.5592],
		"epoch": 20,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.606086, 0.606086, 0.606086, 0.606086],
		"model_val_loss":0.606086,
		"model_train_loss":0.606086
	},
	"ResNet50":{
		"val_loss": [5.009276866912842, 4.370221138000488, 4.016548156738281, 3.797884464263916, 3.6575920581817627, 3.5547163486480713, 3.4766595363616943, 3.418569326400757, 3.369173049926758, 3.3337106704711914, 3.3041746616363525, 3.2746121883392334, 3.2598, 3.2511, 3.2458, 3.2425, 3.2436, 3.2458, 3.2554, 3.2648],
		"train_loss": [6.026840686798096, 4.7727131843566895, 4.218074321746826, 3.915517807006836, 3.7000908851623535, 3.5450735092163086, 3.414228677749634, 3.304046869277954, 3.208782911300659, 3.124225378036499, 3.0475149154663086, 2.977383613586426, 2.9120, 2.8537, 2.7986, 2.7460, 2.6964, 2.6486, 2.6023, 2.5592],
		"epoch": 20,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.606086, 0.606086, 0.606086, 0.606086],
		"model_val_loss":0.606086,
		"model_train_loss":0.606086
	}
}
