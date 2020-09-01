configuration = {
	'dataset': '/content/drive/My Drive/Colab_thesis/Flicker8k_Dataset_new/', # Make sure you put that last slash(/)
    'featuresPath': 'features/', # Make sure you put that last slash(/)
	'modelsPath': 'models/', # Make sure you put that last slash(/)
	'loadModelPath': '',
	'tokenFilePath': 'data/Flickr8k.token.txt',
	'trainImagePath': 'data/Flickr_8k.trainImages.txt',
	'validationImagePath': 'data/Flickr_8k.devImages.txt',
	'CNNmodelType': 'xception', # vgg16/inceptionv3/rasnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 16,
	'maxLength': 40, # This is set manually after training of model and required for test.py
	'beamIndex': 3
}
rnnConfig = {
	'embedding_size': 300,
	'LSTM_GRU_units': 256,
	'dense_units': 256,
	'dropout': 0.5
}
def model_path(modelType):
	if modelType == 'inceptionv3':
		configuration['loadModelPath'] = 'models/model_inceptionv3_GRU_epoch-16_train_loss-2.7460_val_loss-3.2425.hdf5'
	elif modelType == 'vgg16':
		configuration['loadModelPath'] = 'models/model_vgg16_GRU_epoch-16_train_loss-2.7285_val_loss-3.2279.hdf5'
	elif modelType == 'xception':
		configuration['loadModelPath'] = 'models/model_xception_GRU_epoch-15_train_loss-2.7480_val_loss-3.2163.hdf5'
	elif modelType == 'rasnet50':
		configuration['loadModelPath'] = 'models/model_rasnet50_GRU_epoch-15_train_loss-2.7690_val_loss-3.2504.hdf5'

models_summary = {
	"Xception":{
		"val_loss": [4.9473, 4.3192, 3.9664, 3.7579, 3.6171, 3.5152, 3.4394, 3.3801, 3.3352, 3.2943, 3.2667, 3.2495, 3.2314, 3.2204, 3.2163, 3.2196, 3.2250, 3.2320, 3.2428],
		"train_loss": [5.9991, 4.7110, 4.1710, 3.8689, 3.6611, 3.4976, 3.3662, 3.2565, 3.1610, 3.0772, 3.0013, 2.9299, 2.8655, 2.8039, 2.7480, 2.6951, 2.6437, 2.5963, 2.5500],
		"epoch": 15,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.521550, 0.282234, 0.155525, 0.085886],
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
		"Blue_score":[0.528453, 0.285276, 0.158154, 0.087992],
		"model_val_loss":3.2425,
		"model_train_loss":2.7460
	},
	"VGG16":{
		"val_loss": [5.0080156326293945, 4.360769748687744, 3.969210386276245, 3.748412609100342, 3.6370530128479004, 3.5482752323150635, 3.469907760620117, 3.408987283706665, 3.3610379695892334, 3.3194875717163086, 3.288480043411255, 3.2637147903442383, 3.2484023571014404, 3.238137722015381, 3.229300022125244, 3.2279, 3.2303],
		"train_loss": [6.007946968078613, 4.771703243255615, 4.192366123199463, 3.8549320697784424, 3.6583566665649414, 3.5200822353363037, 3.3960094451904297, 3.2855827808380127, 3.1916754245758057, 3.1078412532806396, 3.030799388885498, 2.9607720375061035, 2.897632598876953, 2.836695909500122, 2.780259370803833, 2.7285, 2.6764],
		"epoch": 16,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.523698, 0.287140, 0.161852, 0.091831],
		"model_val_loss":3.2279,
		"model_train_loss":2.7285
	},
	"ResNet50":{
		"val_loss": [4.912473678588867, 4.250346660614014, 3.941431760787964, 3.743990898132324, 3.6250765323638916, 3.5334317684173584, 3.459148406982422, 3.4032039642333984, 3.3577075004577637, 3.320237874984741, 3.293116569519043, 3.272449493408203, 3.258984088897705, 3.250969171524048, 3.2503809928894043, 3.252441883087158],
		"train_loss": [5.9819440841674805, 4.645030498504639, 4.123632431030273, 3.840043306350708, 3.646374225616455, 3.5005416870117188, 3.3719916343688965, 3.26163649559021, 3.1704306602478027, 3.0880143642425537, 3.0128629207611084, 2.9453282356262207, 2.8815176486968994, 2.823620557785034, 2.7690281867980957, 2.719313383102417],
		"epoch": 15,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"Blue_score":[0.533855, 0.288110, 0.159584, 0.085857],
		"model_val_loss": 3.2504,
		"model_train_loss":2.7480
	}
}
