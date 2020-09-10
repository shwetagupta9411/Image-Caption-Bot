configuration = {
	'dataset': 'data/Flicker8k_Dataset/', # Make sure you put that last slash(/)
    'featuresPath': 'features/', # Make sure you put that last slash(/)
	'modelsPath': 'models/', # Make sure you put that last slash(/)
	'loadModelPath': '',
	'tokenFilePath': 'data/Flickr8k.token.txt',
	'trainImagePath': 'data/Flickr_8k.trainImages.txt',
	'validationImagePath': 'data/Flickr_8k.devImages.txt',
	'testImagePath': 'data/Flickr_8k.testImages.txt',
	'CNNmodelType': 'resnet50', # vgg16/inceptionv3/resnet50/xception
	'RNNmodelType': 'GRU', # GRU or LSTM
	'batchSize': 64,
	'epochs': 20,
	'maxLength': 40, # This is set manually after training of model and required for test.py
	'beamIndex': 3
}
rnnConfig = {
	'embedding_size': 256,
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
	elif modelType == 'resnet50':
		configuration['loadModelPath'] = 'models/model_resnet50_GRU_epoch-15_train_loss-2.7690_val_loss-3.2504.hdf5'

models_summary = {
	"Xception":{
		"val_loss": [4.9473, 4.3192, 3.9664, 3.7579, 3.6171, 3.5152, 3.4394, 3.3801, 3.3352, 3.2943, 3.2667, 3.2495, 3.2314, 3.2204, 3.2163, 3.2196, 3.2250, 3.2320, 3.2428],
		"train_loss": [5.9991, 4.7110, 4.1710, 3.8689, 3.6611, 3.4976, 3.3662, 3.2565, 3.1610, 3.0772, 3.0013, 2.9299, 2.8655, 2.8039, 2.7480, 2.6951, 2.6437, 2.5963, 2.5500],
		"val_accuracy": [0.1989416927099228, 0.24471694231033325, 0.29535973072052, 0.31394797563552856, 0.32451409101486206, 0.33287540078163147, 0.3409484028816223, 0.3488687574863434, 0.35315966606140137, 0.35821375250816345, 0.3634713888168335, 0.3663715720176697, 0.3704080581665039, 0.3738170266151428, 0.376801997423172, 0.37851497530937195, 0.38052497530936192, 0.3850, 0.3908],
		"train_accuracy": [0.1531599909067154, 0.22968356311321259, 0.26712724566459656, 0.3015383183956146, 0.3184363543987274, 0.32959073781967163, 0.33909764885902405, 0.3480234444141388, 0.3560746908187866, 0.3628564774990082, 0.3697313666343689, 0.37561604380607605, 0.3814753592014313, 0.38708922266960144, 0.39176931977272034, 0.39677950739860535, 0.39876950739860535, 0.40312, 0.4092],
		"epoch": 15,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"bleu_score_beam":[0.526995, 0.285009, 0.163749, 0.093186],
		"bleu_score_greedy":[0.525471, 0.280804, 0.151211, 0.079574],
		"model_val_loss":3.2163,
		"model_train_loss":2.7480,
		"model_val_acc": 0.3768,
		"model_train_acc":0.3917
	},
	"VGG16":{
		"val_loss": [5.0080156326293945, 4.360769748687744, 3.969210386276245, 3.748412609100342, 3.6370530128479004, 3.5482752323150635, 3.469907760620117, 3.408987283706665, 3.3610379695892334, 3.3194875717163086, 3.288480043411255, 3.2637147903442383, 3.2484023571014404, 3.238137722015381, 3.229300022125244, 3.2279, 3.2303],
		"train_loss": [6.007946968078613, 4.771703243255615, 4.192366123199463, 3.8549320697784424, 3.6583566665649414, 3.5200822353363037, 3.3960094451904297, 3.2855827808380127, 3.1916754245758057, 3.1078412532806396, 3.030799388885498, 2.9607720375061035, 2.897632598876953, 2.836695909500122, 2.780259370803833, 2.7285, 2.6764],
		"val_accuracy": [0.18471218645572662, 0.24892303347587585, 0.2990230917930603, 0.31674638390541077, 0.3276347517967224, 0.3354702889919281, 0.34405210614204407, 0.3517010807991028, 0.35683998465538025, 0.36311522126197815, 0.36852550506591797, 0.3717309534549713, 0.3741223216056824, 0.37559783458709717, 0.3774125576019287, 0.3800, 0.3818],
		"train_accuracy": [0.1451030969619751, 0.22448156774044037, 0.2702924609184265, 0.307329922914505, 0.3224027454853058, 0.3329421579837799, 0.34334051609039307, 0.352393239736557, 0.3595615029335022, 0.36609503626823425, 0.37350592017173767, 0.3792213499546051, 0.3853034973144531, 0.39094841480255127, 0.39563697576522827, 0.4012, 0.4056],
		"epoch": 16,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"bleu_score_beam":[0.528923, 0.289556, 0.166098, 0.095181],
		"bleu_score_greedy":[0.531212, 0.289382, 0.157824, 0.085238],
		"model_val_loss":3.2279,
		"model_train_loss":2.7285,
		"model_val_acc": 0.3800,
		"model_train_acc":0.4012
	},
	"InceptionV3":{
		"val_loss": [5.009276866912842, 4.370221138000488, 4.016548156738281, 3.797884464263916, 3.6575920581817627, 3.5547163486480713, 3.4766595363616943, 3.418569326400757, 3.369173049926758, 3.3337106704711914, 3.3041746616363525, 3.2746121883392334, 3.2598, 3.2511, 3.2458, 3.2425, 3.2436, 3.2458, 3.2554, 3.2648],
		"train_loss": [6.026840686798096, 4.7727131843566895, 4.218074321746826, 3.915517807006836, 3.7000908851623535, 3.5450735092163086, 3.414228677749634, 3.304046869277954, 3.208782911300659, 3.124225378036499, 3.0475149154663086, 2.977383613586426, 2.9120, 2.8537, 2.7986, 2.7460, 2.6964, 2.6486, 2.6023, 2.5592],
		"val_accuracy": [0.18939316272735596, 0.24546317756175995, 0.28884705901145935, 0.31413453817367554, 0.3249211311340332, 0.33662357926368713, 0.3445778489112854, 0.34981852769851685, 0.3564668893814087, 0.36167362332344055, 0.3655065894126892, 0.3691869378089905, 0.3718,  0.3737,  0.3757, 0.3772, 0.3800, 0.3811, 0.3812, 0.3822],
		"train_accuracy": [0.1490807682275772, 0.22648167610168457, 0.26698338985443115, 0.2999303340911865, 0.31843072175979614, 0.3309984505176544, 0.3414927124977112, 0.35026052594184875, 0.3582751154899597, 0.3654264509677887, 0.3719938397407532, 0.37720710039138794, 0.3834, 0.3879, 0.3937, 0.3984, 0.4034, 0.4080, 0.4127, 0.4179],
		"epoch": 16,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"bleu_score_beam":[0.533170, 0.289638, 0.167687, 0.097979],
		"bleu_score_greedy":[0.520312, 0.276614, 0.148699, 0.079103],
		"model_val_loss":3.2425,
		"model_train_loss":2.7460,
		"model_val_acc": 0.3772,
		"model_train_acc":0.3984
	},
	"ResNet50":{
		"val_loss": [4.912473678588867, 4.250346660614014, 3.941431760787964, 3.743990898132324, 3.6250765323638916, 3.5334317684173584, 3.459148406982422, 3.4032039642333984, 3.3577075004577637, 3.320237874984741, 3.293116569519043, 3.272449493408203, 3.258984088897705, 3.250969171524048, 3.2503809928894043, 3.252441883087158],
		"train_loss": [5.9819440841674805, 4.645030498504639, 4.123632431030273, 3.840043306350708, 3.646374225616455, 3.5005416870117188, 3.3719916343688965, 3.26163649559021, 3.1704306602478027, 3.0880143642425537, 3.0128629207611084, 2.9453282356262207, 2.8815176486968994, 2.823620557785034, 2.7690281867980957, 2.719313383102417],
		"val_accuracy": [0.20321562886238098, 0.25245073437690735, 0.30021029710769653, 0.3192903995513916, 0.3300091624259949, 0.33787864446640015, 0.3459177017211914, 0.3529391884803772, 0.35807809233665466, 0.3619449734687805, 0.36586275696754456, 0.37001797556877136, 0.3725789487361908, 0.3746141493320465, 0.37581831216812134, 0.3773277699947357],
		"train_accuracy": [0.15338285267353058, 0.2369505614042282, 0.2754606008529663, 0.30654284358024597, 0.3224619925022125, 0.3355713486671448, 0.3454957902431488, 0.35385453701019287, 0.36147981882095337, 0.36846470832824707, 0.3755144774913788, 0.381438672542572, 0.38638678193092346, 0.3918060064315796, 0.3967569172382355, 0.4011267125606537],
		"epoch": 15,
		"batch_size": 64,
		"optimiser": "Adam",
		"beam_index": 3,
		"bleu_score_beam":[0.535697, 0.288838, 0.161833, 0.089996],
		"bleu_score_greedy":[0.522698, 0.281620, 0.152926, 0.082340],
		"model_val_loss": 3.2504,
		"model_train_loss":2.7480,
		"model_val_acc": 0.3758,
		"model_train_acc":0.3967
	}
}
