# coding: utf-8
#%matplotlib inline

import sys, os
sys.path.append(os.path.abspath("../"))
import numpy as np
from diff_funcs import numerical_gradient
import matplotlib.pyplot as plt

# plt.figure()
# plt.xlim([-2.0, 2.0])
# plt.ylim([-2.0, 2.0])
# plt.grid()
f = lambda x: np.sum(x**2)

# def f(x):
#     if x.ndim == 1:
#         print("action")
#         return np.sum(x**2)
#     else:
#         return np.sum(x**2, axis=1)

# x0 = np.arange(-2.0, 2.1, 0.25)

#         plt.quiver(x0, x1, -dx, -dy,  angles="xy", color="blue")

# plt.draw()
# plt.show()

def hoge(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient(f, x)
        
        return grad

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = hoge(f, np.array([X, Y]) )

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()