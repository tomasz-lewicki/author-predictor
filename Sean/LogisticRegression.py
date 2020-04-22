import numpy as np
import random
# logistic regression class
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out): # n_in = how many features? , n_out = how many classes?
        self.x = input
        self.y = label
        self.W = np.zeros((n_in, n_out))  # initialize W 0
        self.b = np.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))
    
    def softmax(self, x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:  
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
        
    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = self.softmax(np.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)
        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = self.softmax(np.dot(self.x, self.W) + self.b)

        cross_entropy = - np.mean(
            np.sum(self.y * np.log(sigmoid_activation) +
            (1 - self.y) * np.log(1 - sigmoid_activation),
                      axis=1))
        return cross_entropy

        
    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return self.softmax(np.dot(x, self.W) + self.b)

def split80_20(examples):
    sampleIndices = random.sample(range(len(examples)), len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet