import numpy as np

def step_function(x) -> np.array:
    y = x > 0
    return y.astype(np.int)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    x = np.arange(-5.0,5.0,0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()