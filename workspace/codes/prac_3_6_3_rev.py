import numpy as np
import prac_3_4_2 as neuron

network = neuron.init_network()
batch_size = 16
x=np.random.rand(1024,2)

for i in range(0,len(x),batch_size):
    print("===batch", i//16+1, "start===\n")
    x_batch = x[i:i+batch_size]
    y_batch = neuron.forward(network,x_batch)
    print("x :", x_batch, "\ny :", y_batch)
    print("\n===batch", i//16+1, "end===\n")