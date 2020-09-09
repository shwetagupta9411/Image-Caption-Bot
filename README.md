# Image Caption Bot

## Set-up

**Pre-requisites :**
1. A good CPU and a GPU with at least 8GB memory
2. At least 8GB of RAM
3. Active internet to download CNN models (InceptionV3/Xception/VGG16/ResNet50) weights
4. Python3 should be installed
5. Access to colab for training the model (only if your machine is not sufficient)

**Set-up the virtual environment :**
1. virtualenv env-image-caption-bot
2. source env-image-caption-bot/bin/activate
3. pip install -r requirement.txt

## Datset

<strong>Flickr8k Dataset:</strong> <a href="https://forms.illinois.edu/sec/1713398">Dataset Request Form</a>

<strong>UPDATE (April/2019):</strong> The official site seems to have been taken down (although the form still works). Here are some direct download links:

<ul type="square">
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flickr8k_Dataset</a></li>
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flickr8k_text</a></li>
	Download Link Credits:<a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/"> Jason Brownlee</a>
</ul>

<strong>Important:</strong> After downloading the dataset, put the required files in **data** folder

## Configurations (config.py)

**configuration :**

1. **`dataset`** :- Folder path containing flickr dataset images.
2. **`featuresPath`** :- Path for the features folder which will contain all grnerated features pickle files and tokenizer pickle file along with processed captions text file.
3. **`modelsPath`** :- Path to store the trained models.
4. **`tokenFilePath`** :- Filckr8k.token.txt file path containing captions.
5. **`trainImagePath`** :- Flickr_8k.trainImages.txt file path containing image ids for training.
6. **`validationImagePath`** :- Flickr_8k.devImages.txt file path containing image ids for validation.
7. **`testImagePath`** :- Flickr_8k.testImages.txt file path containing image ids for testing (used for evaluation).
8. **`CNNmodelType`** :- CNN Model type to use -> (vgg16/inceptionv3/resnet50/xception).
9. **`RNNmodelType`** :- RNN Model type to use -> (GRU or LSTM).
10. **`batchSize`** :- Batch size for training (larger will consume more GPU & CPU memory).
11. **`epochs`** :- Number of epochs.
12. **`maxLength`** :- Maximum length of captions. This is set manually after training of model and required for testing the images.
13. **`beamIndex`** :- BEAM search parameter which tells the algorithm how many words to consider at a time.

**model_path() :**  *Set models path for testing :*

14. **`model_path()`** :- Function used to set the model path for all the models.
15. **`loadModelPath`** :- Path for all the trained models. Set this variable inside model_path function after training all the models.

**rnnConfig :**

1. **`embedding_size`** :- Embedding size used in Decoder(RNN) Model
2. **`LSTM_GRU_units`** :- Number of LSTM or GRU units in Decoder(RNN) Model
3. **`dense_units`** :- Number of Dense units in Decoder(RNN) Model
4. **`dropout`** :- Dropout probability used in Dropout layer in Decoder(RNN) Model

**`models_summary`** :- update the summary of the models in order to reflect it on the UI.

## Steps to train the models

1. Clone the repository: `git clone https://github.com/shwetagupta9411/Image-Caption-Bot.git`
2. Put the required dataset files in `data` folder -
  - Flickr8k_Datase (folder)
  - Flickr_8k.devImages.txt
  - Flickr_8k.trainImages.txt
  - Flickr_8k.testImages.txt
  - Flickr_8k.token.txt
3. Review `config.py` for paths and other configurations (explained above).
4. Run `train_model.py`.

## Steps to evaluate the models
