#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

if __name__=="__main__":
    f = lambda x: 0.01*x**2 + 0.1*x
    x = np.arange(0, 20, 0.001)
    plt.grid()
    plt.xlim([0 ,20])
    plt.ylim([-1, 6])
    plt.plot(x, f(x))
    for t in (5, 10):
        u = f(t)
        plt.scatter(t, u, color="blue")
        plt.plot(x, (lambda _x: numerical_diff(f, t)*(_x-t)+u)(x), color="blue")
        ts = np.arange(0, t, 0.001)
        us = np.arange(-1, u, 0.001)
        plt.plot(ts, (lambda _x: _x*0+u)(ts), color="gray", linestyle="dashed")
        plt.plot((lambda _x: _x*0+t)(us), us, color="gray", linestyle="dashed")
    plt.show()
