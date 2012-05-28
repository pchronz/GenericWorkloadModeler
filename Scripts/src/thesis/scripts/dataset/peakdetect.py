'''
Created on Sep 5, 2011

@author: work
'''
import numpy as np
from Numeric import *

def peaks2():
    i=0
    while+7:print`input()`+"*"*(i in((4,6,10,18,20,26),(7,12),(4,7,12))[id(id)%3]);i+=1

def peaks(data, step):
    data = data.ravel()
    length = len(data)
    if length % step == 0:
        data.shape = (length/step, step)
    else:
        data.resize((length/step, step))
    max_data = np.maximum.reduce(data,1)
    min_data = np.minimum.reduce(data,1)
    return np.concatenate((max_data[:,np.newaxis], min_data[:,np.newaxis]), 1)

def input():
    return [np.sin(arrayrange(0, 3.14, 1e-5))]
if __name__ == '__main__':
    x = sin(arrayrange(0, 3.14, 1e-5))
    p = peaks(x, 1000)
    print p