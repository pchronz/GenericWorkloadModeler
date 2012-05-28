'''
Created on Jul 14, 2011

@author: work
'''


from heapq import nlargest
from numpy.fft import fft
from numpy import exp
from pylab import find
from pymc import *
from random import randint
from thesis.scripts.bayesian.poisson import poisson_req, sme_calc
import sys
import time
def main():
    from thesis.scripts.bayesian import NewRequestmodel
    model = MCMC(NewRequestmodel)
    
    starttime = time.time()
    model.sample(iter=1000, burn=200, thin=10)
    print "Training time"
    print time.time() - starttime
    
    testinput = []
    for i in range (200):
        testinput.append(randint(0,1007))
    
    for i in range(10):
        reqs = poisson_req(model, testinput, i)
        
        ttarget = []
        for prob in reqs:
    
            m = max(prob)
            el = find(prob > m*2/3)
            maxes = nlargest(15, prob)
            maxvals = [prob.index(maxval) for maxval in maxes]
            ttarget.append(maxvals)
        
        sme = sme_calc(ttarget, model, i)
        print "SME = %f" % sme
    return reqs

if __name__ == '__main__':
    main()