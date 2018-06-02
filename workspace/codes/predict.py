# coding: utf-8
import numpy as np
from act_funcs import sigmoid, relu, identity_function

# `network` must have attributes like these:

# - "layer_num": int
# - "W": list of weights like [n["W1"], n["W2"], n["W3"]]
# - "B": list of biases like [n["b1"], n["b2"], n["b3"]]
class Predict:

    def __init__(self, actf, outf):
        self.actf = actf
        self.outf = outf

    def _predict(self, n, i):
        a = np.dot(n["X"],n["W"][i]) + n["B"][i]
        n["X"] = self.actf(a)
        i += 1
        if i < n["layer_num"]:
            return self._predict(n, i)
        else:
            return self.outf(a)

    def predict(self, n, x):
        n["X"] = x
        n["layer_num"] = len(n["W"])
        return self._predict(n, 0)