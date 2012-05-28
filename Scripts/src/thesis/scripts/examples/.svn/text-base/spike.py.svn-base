'''
Created on Sep 7, 2011

@author: work
'''
from numpy import array, maximum, minimum, concatenate, newaxis
from scipy import signal, diff, split
from thesis.scripts.dataset.dataset import weeklydataset_sg_ndata
from thesis.scripts.samples.aggregatesamples import aggregatebymins_sg_ndata
import itertools
import pylab
import time

def detect_signals():
    vector, label = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/access_log-20110805.csv',[])
    x, target = aggregatebymins_sg_ndata(vector[1])
    
    starttime = time.time()
    y = array(target)
    t = array(x)
    thr = max(y)* 2/3
    print thr
    I = pylab.find(y > thr)
#    print I
#    pylab.plot(t,y, 'b',label='signal')
#    pylab.plot(t[I], y[I],'ro',label='detections')
#    pylab.plot([0, t[len(t)-1]], [thr,thr], 'g--')
    
    J = pylab.find(diff(I) > 1)
    argpeak = []
    targetpeak = []
    for K in split(I, J+1):
        ytag = y[K]
        peak = pylab.find(ytag == max(ytag))
#        pylab.plot(peak+K[0],ytag[peak],'sg',ms=7)
        argpeak.append(peak+K[0])
        targetpeak.append(ytag[peak])
    
    eta = time.time() - starttime
    print "time elapsed %f" % eta
    return list(itertools.chain(*argpeak)), list(itertools.chain(*targetpeak))



def peaks():
    vector, label = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/access_log-20110805.csv',[])
    x, target = aggregatebymins_sg_ndata(vector[1])
    
    data = array(target)
    t = array(x)
    data = data.ravel()
    length = len(data)
    print length
    step = 40
    if length % step == 0:
        data.shape = (length/step, step)
    else:
        data.resize((length/step, step))
    max_data = maximum.reduce(data,1)
    min_data = minimum.reduce(data,1)
    
    pylab.plot(t,array(target), 'b',label='signal')
    return concatenate((max_data[:,newaxis], min_data[:,newaxis]), 1)
