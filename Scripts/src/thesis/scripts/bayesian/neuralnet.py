'''
Created on Jul 13, 2011

@author: Claudio
'''
import sys
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from types import *

def bayesian_net(train, train_label, layers):
    
    inputs = 0
    targets = 0
    print train_label[0]
    if type(train[0]) is list:
        inputs = len(train[0])
    else:
        inputs = 1
    
    if type(train_label[0]) is list:
        targets = len(train_label[0])
    else: targets = 1
        
    
    ds = SupervisedDataSet(inputs,targets)
    
    for i in range(len(train)):
        ds.addSample(train[i], train_label[i])
        
    net = buildNetwork(inputs, layers, targets, bias=True, hiddenclass=SigmoidLayer)
    
    trainer = BackpropTrainer(net, ds)
    
    trainer.trainUntilConvergence()
    
    return trainer