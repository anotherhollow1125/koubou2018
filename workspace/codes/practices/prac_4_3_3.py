import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        print(idx)
        f1,f2 = (f(np.array([*x[:idx], x[idx]+d, *x[idx+1:]])) for d in (h, -h))
        grad[idx] = (f1 - f2) / (2*h)
    
    return grad

if __name__=="__main__":
    f = lambda x: x[0]**2+x[1]**2
    print(numerical_gradient(f, np.array([3.0, 4.0])))
    print(numerical_gradient(f, np.array([0.0, 2.0])))
    print(numerical_gradient(f, np.array([3.0, 0.0])))
