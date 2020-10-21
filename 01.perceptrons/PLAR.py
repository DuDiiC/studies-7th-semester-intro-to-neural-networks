# ------------------------------------------
# Maciej Dudek
# 20.10.2020
# ------------------------------------------

import numpy as np

class PLAR:
    def __init__(self, label, size, learning_set, answers, learning_rate = 0.1):
        self.label = label
        self.size = size
        self.learning_set = learning_set
        self.answers = answers
        self.learning_rate = learning_rate
        self.weights = np.random.randn(size + 1) # add theta into vector w
        self.best_weights = self.weights
        self.best_lifetime = 0
        self.best_predict_number = 0
        self.current_lifetime = 0

    def train(self, iterations):
        for i in range(iterations):
            tmp_i = np.random.randint(0, self.learning_set.shape[0])
            tmp_j = np.random.randint(0, self.learning_set.shape[1])
            predict = self.forward(self.learning_set[tmp_i][tmp_j])
            if self.answers[tmp_i * tmp_j] - predict != 0:
                self.weights[1:] += self.learning_rate * (self.answers[tmp_i * tmp_j] - predict) * self.learning_set[tmp_i][tmp_j]
                self.weights[0] += self.learning_rate * (self.answers[tmp_i * tmp_j] - predict)
                self.current_lifetime = 0
            else:
                self.current_lifetime += 1
                current_predict_number = self.current_predict_number()
                if(self.current_lifetime > self.best_lifetime & current_predict_number > self.best_predict_number):
                    self.best_predict_number = current_predict_number
                    self.best_lifetime = self.current_lifetime
                    if (self.best_weights != self.weights).all():
                        self.best_weights = np.copy(self.weights)

    def forward(self, data):
        """Calculates the main part of the algorithm"""
        dot = np.sum(data * self.weights[1:]) + self.weights[0]
        return self.activ_func(dot)

    def forward_predict(self, data):
        dot = np.sum(data * self.best_weights[1:]) + self.best_weights[0]
        return self.activ_func(dot)

    def activ_func(self, dot):
        """Activation fuction"""
        if dot > 0:
            return 1
        else:
            return -1

    def predict(self, data):
        """A function that returns the response proposed by the perceptron for the given data"""
        return self.forward_predict(data)

    def current_predict_number(self):
        correct_number = 0
        for i in range(10):
            for j in range(5):
                if self.forward(self.learning_set[i][j] == self.answers[i * j]):
                    correct_number += 1
        return correct_number

