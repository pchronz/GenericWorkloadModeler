'''
Created on Jul 20, 2011

@author: work
'''
from thesis.scripts.clustering.clustering import create_clustered_samples
from thesis.scripts.samples.aggregatesamples import aggregatebymins_sg
from thesis.scripts.dataset.dataset import weeklydataset_shogun
from thesis.scripts.svr.svr import SVR
from thesis.scripts.samples.traintest import traintest
from matplotlib.pyplot import figure, show
from numpy import array, int32
import time

def svrmodel(traininput, traintarget, testinput, testtarget):
#    [points, label] = weeklydataset_shogun(filesource, [])
#    clusteredpoints, cdata = create_clustered_samples(points, 5, 1)
#    cluster0 = clusteredpoints[0]
#    input, target = aggregatebymins_sg(cluster0[0])
##    testinput, testtarget = aggregatebymins_sg(cluster0[0])
#    print len(input)
#    traininput, traintarget, testinput, testtarget = traintest(input, target, 20, 1)
    svr = SVR(traininput, testinput, traintarget,2,256,0.1,0.5)
    out = svr.svr_req(testinput)
    sme = svr.calc_sme(testtarget, out)
    x = array(testinput, dtype=int32)
    y = array(testtarget, dtype=int32)
    xp = array(testinput, dtype=int32)
    yp = array(out, dtype=int32)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, y)
    ax1.plot(xp,yp,"r")
    ax1.axis([0,max(x)+10,0,max(y)+10])
    ax1.set_xlabel('minutes of the week')
    ax1.set_ylabel('number of requests')
    fig.savefig("svr_model_%f" % (time.time()), format='png')
    
    print "SME = %f" % sme

