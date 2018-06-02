# coding: utf-8
import sys, os
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../../deep-learning-from-scratch-master/"))
import numpy as np
from act_funcs import sigmoid, relu, identity_function
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def _init_network():
    with open("../sample_weight.pkl", "rb") as f:
        _network = pickle.load(f)
    
    return _network

def init_network(layer_num):
    _network = _init_network()
    network = {}
    network["layer_num"] = layer_num
    network["B"] = np.array([_network["b"+str(i)] for i in range(1,4)])
    network["W"] = np.array([_network["W"+str(i)] for i in range(1,4)])

    return network

def _predict(n,i):
    a = np.dot(n["X"],n["W"][i]) + n["B"][i]
    n["X"] = sigmoid(a)
    i += 1
    if i < n["layer_num"]:
        return _predict(n,i)
    else:
        return identity_function(a)

def predict(n, x):
    n["X"] = x
    return _predict(n, 0)

def main():
    x, t = get_data()
    network = init_network(3)

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
    print("Accuracy:", float(accuracy_cnt)/len(x))

if __name__=="__main__":
    main()