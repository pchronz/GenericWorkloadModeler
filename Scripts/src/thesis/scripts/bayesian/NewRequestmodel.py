'''
Created on Aug 28, 2011

@author: work
'''
from pymc import *
from thesis.scripts.examples.main_ex import main_mcmc
from thesis.scripts.examples.article_main import initialize_wmproxy
from numpy import matrix, int32
import numpy

#target of the Barnes-Hut
#target = main_mcmc()

 #initialization of data wmproxy
traininput, traintarget, testinput, testtarget = initialize_wmproxy()
target = [traintarget]


rates = []
requests = []
for j in range(len(target)):
    val = matrix(target[j], dtype = int32)
    #prior
    rates.append([Normal('rate_%d_%d' % (j,i),mu = numpy.mean(val[i]), tau = 1.0, value = 1) for i in range(len(val))])
    
    #likelihood
    requests.append([Poisson('request_%d_%d' % (j,i), mu=rates[j][i], value=val[i], observed=True) for i in range(len(val))])