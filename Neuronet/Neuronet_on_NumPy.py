
import random as rnd
import numpy as np

def sigmoid_func(par):
    return 1/(1 + np.exp(-par))

def randlist(par):
    out = []
    for i in range(par):
        out.append(rnd.random())
    return out

def der_sigmoid_func(par):
    return (sigmoid_func(par)) * (1 - sigmoid_func(par))

class neuronet():
    
    def __init__(self, L, sizes):
        self.layerscount = L #scalar
        self.sizes = sizes #array

        self.b = []#смещения(вектор)
        self.W = []#веса каждого слоя(трёхмерный тензор)
        self.dW = []#производные весов слоя(трёхмерный тезор)
        self.dB = []#производные смещений(вектор)
        self.z = []#Суммы
        for i in range(self.layerscount - 1):
            self.W.append(np.random.random((self.sizes[i+1], self.sizes[i])))
            self.b.append(np.random.random((self.sizes[i+1])))

        self.W = np.array(self.W)
        self.b = np.array(self.b)


    def forward(self, input):
        #input-vector

        self.z = []
        output = input
        self.z.append(output)
        for i in range(self.layerscount-1):
            output = self.W[i].dot(output) + self.b[i]
            self.z.append(output)
            output = sigmoid_func(output)
        self.z = np.array(self.z)
        return output

    def backward(self, y, out):

        #Расчитываем дельты слоёв(дельты смещений)
        self.dW = []#производные весов слоя(трёхмерный тезор)
        self.dB = []
        d1 = (out - y) * der_sigmoid_func(self.z[self.layerscount-1])
        d = d1
        self.dB.append(d)
        for i in range(self.layerscount-1, 0, -1):
            tr = np.transpose(self.W[i-1])
            d = tr.dot(d) * der_sigmoid_func(self.z[i-1])
            self.dB.append(d)
        #self.dB.reverse()
        self.dB = np.array(self.dB)
        #Расчитываем дельты весов
        #d =np.dot(d1[:,np.newaxis], np.transpose(self.z[self.layerscount-2][:,np.newaxis]))
        #self.dW.append(d)
        for i in range(self.layerscount-1):
            d = np.dot(self.dB[self.layerscount-2-i][:,np.newaxis],np.transpose(sigmoid_func(self.z[i][:,np.newaxis])))
            self.dW.append(d)
        self.dW = np.array(self.dW)

    def train(self):

        train_arr1 = np.array([0,1,1,0])
        train_arr2 = np.array([0,0,0,0])

        in_arr1 = np.array([0,0,0,1,1,0,1,1])
        in_arr2 = np.array([0,0,0,0,0,0,0,0])

        prec = 0.2
        out = self.forward(np.array(in_arr1))

        for i in range(20000):
            out = self.forward(in_arr1)
            self.backward(train_arr1, out)
            self.changeW(1.2)
            out = self.forward(in_arr2)
            self.backward(train_arr2, out)
            self.changeW(1.2)
            #print("%d once ", i)
            #print(out)
        print(self.forward(in_arr1))
        print(self.forward(in_arr2))
        print(self.forward(np.array([1,1,1,1,1,1,1,1])))
        

    def changeW(self, alpha):
        m = self.sizes[self.layerscount-1]

        for k in range(self.layerscount-1):
           self.W[k] += (-1*alpha/m) * self.dW[k]
           self.b[k] += (-1*alpha/m) * self.dB[self.layerscount-2-k]
            

        
nn = neuronet(4,[8, 6, 6, 4])#(num_of_layers,[n_of_neurons_on_l0, n_of_neurons_on_l1, n_of_neurons_on_l2, ... n_of_neurons_on_ln-1])



nn.train()

