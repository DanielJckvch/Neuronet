
import random as rnd
import numpy as np

def sigmoid_func(par):
    return 1/(1 + np.exp(-par))

def randlist(par):
    out = []
    for i in range(par):
        out.append(rnd.random())
    return out

class neuronet():
    
    def __init__(self, L, sizes):
        self.layerscount = L #scalar
        self.sizes = sizes #array
        self.layers = []
        c = []
        for k in range(self.layerscount-1):
            b = []
            for i in range(self.sizes[k+1]):
                a = []
                for j in range(self.sizes[k]):
                    a = randlist(sizes[k])
                b.append(a)
            c.append(b)
        self.layers = c
        print(c)

    def forward(self, input):
        #input-vector
        #output[0][0] = 0
        output = [input[:]]

        for k in range(self.layerscount-1):
            a=[]
            for i in range(self.sizes[k+1]):
                z = 0
                
                for j in range(self.sizes[k]):
                    z += output[k][j] * self.layers[k][i][j]
                a.append(sigmoid_func(z))
            output.append(a)
        return output

nn = neuronet(3,[6,6,4])#(num_of_layers,[n_of_neurons_on_l0, n_of_neurons_on_l1, n_of_neurons_on_l2, ... n_of_neurons_on_ln-1]

out = nn.forward([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,])

print(out)