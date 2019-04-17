# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 00:52:13 2019

@author: Hossein
"""
import numpy as np  
import matplotlib.pyplot as plt
feature_set = np.array([[0,1],[0,0],[1,0],[1,1]])  
labels = np.array([[1,0,1,1]])  
labels = labels.reshape(4,1)  
np.random.seed(42)  
weights = np.random.rand(2,1)  
bias = np.random.rand(1)  
lr = .1
def sigmoid(x):  
    return 1/(1+np.exp(-x))
def sigmoid_der(x):  
    return sigmoid(x)*(1-sigmoid(x))
x1=np.empty(0)
epochs=20000
for epoch in range(epochs): 
    
    inputs = feature_set
    XW = np.dot(feature_set, weights) + bias

    z = sigmoid(XW)


    error = z - labels
    eer=np.power(error,2)
    print(error.sum())
    er=error.sum()
    
    x1=np.append(x1,er)

    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    dw=np.dot(inputs, z_delta)
    weights -= lr * dw

#    for num in z_delta:
    bias -= lr * z_delta.mean()
def pri(x):
    pri1=sigmoid(np.dot(x,weights)+bias)
    print(pri1)
    return
#pri([1,1])
plt.plot(np.arange(1,epochs+1,1),x1,)
plt.show()


    