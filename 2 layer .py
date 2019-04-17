# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:31:22 2019

@author: Hossein
"""
import numpy as np
import matplotlib.pyplot as plt
inputs=np.array([[1,1,0,0],[1,0,1,0]]).T
target=np.array([[0,1,1,0]]).T
epochs=20000
learningRate=1
weightInput=np.random.random((2,2))
weightOutput=np.random.random((2,1))
def sigmoid(inp,der=False):
    segx=1/(1+np.exp(-inp))
    if der==True:
        return inp*(1-inp)
    else:
        return segx
def backpropagation (error,output,weight):
    dz=np.multiply(sigmoid(output,der=True),error)
    bp=np.dot(dz,weight)
    return bp
def deltaweight(error,output,prelayer):
    
    dz=np.multiply(sigmoid(output,der=True),error)
    dw=np.dot(prelayer,dz)
    return dw
def feedforward(inputs,weightInput,weightOutput):
    outputLayer1=sigmoid(np.dot(inputs,weightInput))
    outputLayer2=sigmoid(np.dot(outputLayer1,weightOutput))
    return outputLayer1,outputLayer2

def error(outputs,target):
    er=(target-outputs)
    return er
y=np.empty(0)
for i in range(epochs):
    outputLayer1,outputLayer2=feedforward(inputs,weightInput,weightOutput)

    error1=error(outputLayer2,target)
    eer=np.power(error1,2)
    y=np.append(y,eer.sum())
    print(eer.sum())
    deltaWeightOutput=deltaweight(error1,outputLayer2,outputLayer1.T)
    weightOutput+=learningRate*deltaWeightOutput 
    deltaWeightInput=deltaweight(backpropagation(error1,outputLayer2,weightOutput.T),outputLayer1,inputs.T)
    
    weightInput+=learningRate*deltaWeightInput

plt.plot(np.arange(0,epochs,1),y,color='blue')
plt.ylabel("Error")
plt.xlabel("epochs")
plt.title("mlp neural network")
#plt.yticks([0.05],"the best error" )
def pri(inputs):
    global weightInput
    global weightOutput
    out1,out2=feedforward(inputs,weightInput,weightOutput)
    print (out2)
    
    
    


    

    
    
    

