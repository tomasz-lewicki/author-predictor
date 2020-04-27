import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re


def EUdist(v1, v2):
    s = 0
    for i, j in zip(v1,v2):
        s += abs(i-j)
    return s**0.5

def Consine_dissimilarity(v1, v2):
    upp = np.dot(v1,v2)
    lower = (np.sum(np.square(v1)) * np.sum(np.square(v1)))**0.5
    return 1-upp/lower

def dissimilarity(clusters):
    """Assumes clusters a list of clusters
       Returns a measure of the total dissimilarity of the
       clusters in the list"""
    totDist = 0
    for c in clusters:
        totDist += c.variability()
    return totDist

def clean_text(series): 
	''' input: Series or np.array
		output: cleaned text in a 2D array'''
	stop_words = set(stopwords.words('english')) 
	lines = []
	for i in series:
	    line = word_tokenize(i)
	    filtered_sentence = [w.lower() for w in line if not w in stop_words]
	    filtered_sentence = [re.sub(r'\W+','',i) for i in filtered_sentence]
	    filtered_sentence = [re.sub(r'\d+','',i) for i in filtered_sentence]
	    filtered_sentence = [i for i in filtered_sentence if i]
	    lines.append(filtered_sentence)


	return lines


class Example(object):
    
    def __init__(self, name, features, label = None, line=[]):
        #Assumes features is an array of floats
        self.name = name
        self.features = features
        auth = ['EAP', 'HPL','MWS']

        if label == auth[0]:
            self.label = [1,0,0]
        elif label == auth[1]:
        	self.label = [0,1,0]
        elif label == auth[2]:
        	self.label = [0,0,1]
        else:
            self.label = None
        self.text = line
        
    def dimensionality(self):
        return len(self.features)
    
    def getFeatures(self):
        return self.features[:]
    
    def getLabel(self):
        return self.label
    
    def getName(self):
        return self.name
    
    def distance(self, other):
        return Consine_dissimilarity(self.features, other.getFeatures())
    
    def __str__(self):
        return str(self.name) + ':' + str(self.features) + ':'+ str(self.label)
