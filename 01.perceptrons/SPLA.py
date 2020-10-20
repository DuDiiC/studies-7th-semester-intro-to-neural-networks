# ------------------------------------------
# Maciej Dudek
# 14.10.2020
# ------------------------------------------

import numpy as np

class SPLA:
    def __init__(self, label, size, learning_rate = 0.1):
        self.label = label
        self.size = size
        self.learning_rate = learning_rate
        self.weights = np.random.randn(size + 1) # add theta into vector w

    def train(self, data, answer):
        predict = self.forward(data)
        if answer - predict != 0:
            self.weights[1:] += self.learning_rate * (answer - predict) * data
            self.weights[0] += self.learning_rate * (answer - predict)

    def forward(self, data):
        """Calculates the main part of the algorithm"""
        dot = np.sum(data * self.weights[1:]) + self.weights[0]
        return self.activ_func(dot)

    def activ_func(self, dot):
        """Activation fuction"""
        if dot >= 0:
            return 1
        else:
            return -1

    def predict(self, data):
        """A function that returns the response proposed by the perceptron for the given data"""
        return self.forward(data)