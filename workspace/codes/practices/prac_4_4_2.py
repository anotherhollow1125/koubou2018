# coding: utf-8

import os, sys
sys.path.append(os.path.abspath("../"))
import numpy as np
from loss_funcs import cross_entropy_error
from diff_funcs import numerical_gradient
from act_funcs import softmax

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

if __name__=="__main__":
    pass
    # import numpy as np
    # import prac_4_4_2 as p4
    # net = p4.simpleNet()
    # print(net.W)
    # x = np.array([0.6, 0.9])
    # p = net.predict(x)
    # print(p)
    # np.argmax(p)
    # t = np.array([0, 0, 1]) # 例です。
    # net.loss(x, t)
    # f = lambda _: net.loss(x, t)
    # dW = p4.numerical_gradient(f, net.W)