import os
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
nltk.download('punkt')

ERROR_THRESHOLD = 0.25
class Prediction:
    # This class is mainly used to predict the answer from user's question. 
    # It need below informations:
    # - data.pickle file which store all preprocessing data from training step
    # - input file or intent file (its name is stored as part of data.pickle file)
    # - model.tflearn - training model from training step
    context = {}

    def load_model(self):
        if hasattr(self, 'data'):
            return

        with open("{}/data.pickle".format(self.MODEL_DIR), "rb") as f:
            self.words, self.labels, self.training, self.output, self.input_file = pickle.load(f)

        with open(self.input_file) as file:
            self.data = json.load(file)
        
        tensorflow.reset_default_graph()
        tflearn.init_graph(num_cores=1)

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)
        self.model = tflearn.DNN(net)
        self.model.load("{}/model.tflearn".format(self.MODEL_DIR))

    def __init__(self, model_dir = '.'):
        self.MODEL_DIR = model_dir if model_dir else '.'
        self.words = []
        self.labels = []
        self.docs_x = []
        self.docs_y = []
        
        try:
            self.load_model()
        except Exception as e:
            print('Failed to load model, please train it first. Error {}'.format(e))
    
    def bag_of_words(self, s):
        bag = [0 for _ in range(len(self.words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        print('questions: {}'.format(s_words))
        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    bag[i] = 1
                
        return numpy.array(bag)

    def classify(self, sentence):
        # generate probabilities from the model
        results = self.model.predict([self.bag_of_words(sentence)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.labels[r[0]], r[1]))
        # return tuple of intent and probability
        print('classify: ', return_list)
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        print('Initial context:', json.dumps(Prediction.context, indent = 4))
        randomResponse = '' 
        #if not userID in Prediction.context:
        #     #Prediction.context[userID] = ''
                #Prediction.context.update( {userID : ''} )
        #     print('First time I see you!!!')
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                count = 0
                for i in self.data["intents"]:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # if ('context_filter' in i):
                        #     print('UserID', userID, Prediction.context[userID], i['context_filter'])
                        # if (not 'context_filter' in i): 
                        #     print('no context filter')
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            # if not userID in Prediction.context:
                            #     Prediction.context.update( {userID : i['context_set']} )
                            Prediction.context.update({userID: i['context_set']})
                            print ('Set new context: ', Prediction.context[userID])
                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in Prediction.context and 'context_filter' in i and i['context_filter'] == Prediction.context[userID]):
                            print('current try: ', count, results)
                            if show_details: print ('tag:', i['tag'])
                            if show_details: randomResponse += random.choice(i['responses'])
                            if show_details: randomResponse+="\ncurrent context:"
                            if show_details: randomResponse+=json.dumps(Prediction.context, indent = 4)
                            if show_details: randomResponse+="\nanswer tag: "
                            if show_details: randomResponse+=i['tag']
                            if show_details: randomResponse+="\nActual userID: "
                            if show_details: randomResponse+=userID
                            #randomResponse+=results[0][0]
                            # a random response from the intent
                            return randomResponse
                count+=1
                results.pop(0)
        if randomResponse == '':
            randomResponse+='Hello, sunshine :sunflower:! How are you?'
            if show_details: randomResponse+="\ncurrent context:"
            if show_details: randomResponse+=json.dumps(Prediction.context, indent = 4)
            #if show_details: randomResponse+="\nanswer tag: "
            #if show_details: randomResponse+=i['tag']
            if show_details: randomResponse+="\nActual userID " 
            if show_details: randomResponse+=userID
            return randomResponse