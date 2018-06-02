import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# def _numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     for idx in range(x.size):
#         f1,f2 = (f(np.array([*x[:idx], x[idx]+d, *x[idx+1:]])) for d in (h, -h))
#         grad[idx] = (f1 - f2) / (2*h)
    
#     return grad

# def numerical_gradient(f, X):
#     if X.ndim == 1:
#         return _numerical_gradient(f, X)
#     else:
#         grad = np.zeros_like(X)
        
#         for idx, x in enumerate(X):
#             grad[idx] = _numerical_gradient(f, x)
        
#         return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        _x = x[idx]
        f1, f2, _ = ([exec("x[idx]=d", {"x": x, "idx": idx, "d": d}), f(x)][1] for d in (_x+h, _x-h, _x))
        grad[idx] = (f1 - f2) / (2*h)
        it.iternext()
        
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num): x -= lr * numerical_gradient(f, x)

    return x
