import numpy as np
from act_funcs import sigmoid, relu, identity_function

def init_network():
    network = {}
    # network["X"] = np.array([1.0, 0.5])
    network["layer_num"] = 3 # 層の数
    network["B"] = np.array([[0.1, 0.2, 0.3]
                            ,[0.1, 0.2]
                            ,[0.1, 0.2]
    ])

    network["W"] = np.array([
        [[0.1, 0.3, 0.5]
        ,[0.2, 0.4, 0.6]]

        ,
        [[0.1, 0.4]
        ,[0.2, 0.5]
        ,[0.3, 0.6]]

        ,
        [[0.1, 0.3]
        ,[0.2, 0.4]]
    ])

    return network

def _forward(n,i):
    a = np.dot(n["X"],n["W"][i]) + n["B"][i]
    n["X"] = sigmoid(a)
    i += 1
    if i < n["layer_num"]:
        return _forward(n,i)
    else:
        return identity_function(a)

def forward(n, x):
    n["X"] = x
    return _forward(n, 0)

def main():
    print(forward(init_network(), np.array([1.0, 0.5])))

if __name__=="__main__":
    main()