'''
Created on May 28, 2012

@author: claudio
'''

'''
Created on Aug 28, 2011

@author: work
'''
from pymc import *
#from thesis.scripts.examples.main_ex import main_mcmc
from thesis.scripts.examples.article_main import initialize_wmproxy
from numpy import matrix, int32
import numpy

#target of the Barnes-Hut
#target = main_mcmc()

 #initialization of data wmproxy
traininput, traintarget, testinput, testtarget = initialize_wmproxy()

for i in range(len(traintarget)):
        if(traintarget[i] != 0):
            traintarget[i] = numpy.log(traintarget[i])

for i in range(len(testtarget)):
        if(testtarget[i] != 0):
            testtarget[i] = numpy.log(testtarget[i])

rates = []
requests = []
for j in range(len(traintarget)):
    #prior
    rates.append(Normal('rate_%d' % (j), mu = traintarget[j], tau = 1.0, value = 1))
    
    #likelihood
    requests.append(Poisson('request_%d' % (j), mu=rates[j], value=traintarget[j], observed=True))