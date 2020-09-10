from tensorflow.keras.models import load_model
from config import configuration, model_path
from utils import Utils
from pickle import load
from tqdm import tqdm



class evaluate(object):
    def __init__(self, utils):
        self.utils = utils

    def searchEvaluation(self, model, images, captions, tokenizer):
        actual, predicted, predicted_greedy = list(), list(), list()
        for image_id, caption_list in tqdm(captions.items()):
            yhat = self.utils.beamSearchCaptionGenerator(model, images[image_id], tokenizer)
            greedy = self.utils.greedySearchCaptionGenerator(model, images[image_id], tokenizer)
            ground_truth = [caption.split() for caption in caption_list]
            actual.append(ground_truth)
            predicted.append(yhat.split())
            predicted_greedy.append(greedy.split())
        print('Cumulative N-Gram BLEU Scores :')
        print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.\n')
        print("For Beam Search")
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

        print("For Greedy Search")
        print('BLEU-1: %f' % corpus_bleu(actual, predicted_greedy, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted_greedy, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted_greedy, weights=(0.33, 0.33, 0.33, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted_greedy, weights=(0.25, 0.25, 0.25, 0.25)))

    def start(self):
        """ Evaluates the model on training data and output BLEU score """
        # Please set the models path you want to evaluate else comment the code
        print("------------- Available test data -------------")
        testCaptions, testFeatures = self.utils.dataLoader(configuration['testImagePath'])
        print("Calculating BLEU score on testing set for BEAM Search and Greedy Search both")
        model_path(configuration['CNNmodelType'])
        print("evaluation model: ", configuration['loadModelPath'])
        evaluate_model = load_model(configuration['loadModelPath'])
        tokenizer = load(open(configuration['featuresPath']+'tokenizer.pkl', 'rb'))

        self.searchEvaluation(evaluate_model, testFeatures, testCaptions, tokenizer)

if __name__ == '__main__':
    utils = Utils()
    evaluate = evaluate(utils)
    evaluate.start()
