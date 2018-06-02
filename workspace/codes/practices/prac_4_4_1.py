# coding: utf-8
import os, sys
sys.path.append(os.path.abspath("../"))
import numpy as np
from diff_funcs import numerical_gradient, gradient_descent
import matplotlib.pyplot as plt

r = []
f = lambda x: x[0]**2+x[1]**2
init_x = np.array([-3.0, 4.0])
for i in range(1,101):
    xy = gradient_descent(f, init_x, lr=0.01, step_num=i)
    r.append([xy[0], xy[1]])
# plt.xlim([-5, 5])
# plt.ylim([-5, 5])
plt.grid
plt.scatter(*zip(*r))
plt.show()
