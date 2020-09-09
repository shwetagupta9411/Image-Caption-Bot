# Image Caption Bot

## Set up
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

<strong>Flickr8k Dataset:</strong> <a href="https://forms.illinois.edu/sec/1713398">Dataset Request Form</a>

<strong>UPDATE (April/2019):</strong> The official site seems to have been taken down (although the form still works). Here are some direct download links:

<ul type="square">
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flickr8k_Dataset</a></li>
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flickr8k_text</a></li>
	Download Link Credits:<a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/"> Jason Brownlee</a>
</ul>

<strong>Important:</strong> After downloading the dataset, put the reqired files in *data* folder
