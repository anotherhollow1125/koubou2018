import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

# def softmax(a):
#     exp_a = np.exp(a - max(a))
#     return exp_a/np.sum(exp_a)
def softmax(x):
    if x.ndim >= 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

if __name__=="__main__":
    # import pandas as pd
    import matplotlib.pyplot as plt
    x = np.arange(-5,5,0.01)
    plt.grid()
    plt.plot(x,sigmoid(x))
    plt.show()
    plt.grid()
    plt.plot(x,relu(x))
    plt.show()
