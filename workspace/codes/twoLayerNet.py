# coding: utf-8
import numpy as np
# import os, sys
# sys.path.append(os.path.abspath("../"))
from predict import Predict
from loss_funcs import cross_entropy_error
from diff_funcs import numerical_gradient
from act_funcs import sigmoid, softmax

class TwoLayerNet:

    def __init__(self, input_size,
            hidden_size, output_size,
            weight_init_std=0.01):
        n = {}
        n["W"] = [weight_init_std * np.random.randn(s1, s2)
            for s1, s2 in ((input_size, hidden_size),(hidden_size, output_size))]
        n["B"] = [np.zeros(s) for s in (hidden_size, output_size)]
        self._net = n
        self._p = Predict(sigmoid, softmax)
    
    @property
    def params(self):
        d = {"W1": self._net["W"][0]
            ,"W2": self._net["W"][1]
            ,"b1": self._net["B"][0]
            ,"b2": self._net["B"][1]}
        return d
    
    def predict(self, x):
        return self._p.predict(self._net, x)
    
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        return {i: numerical_gradient(loss_W, d)
            for i, d in (("W1", self._net["W"][0])
                        ,("b1", self._net["B"][0])
                        ,("W2", self._net["W"][1])
                        ,("b2", self._net["B"][1]))}
        