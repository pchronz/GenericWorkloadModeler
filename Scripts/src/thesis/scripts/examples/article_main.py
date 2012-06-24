'''
Created on May 21, 2012

@author: claudio
'''
from heapq import nlargest
from matplotlib import pylab
import numpy
from matplotlib.pyplot import figure, show
from numpy import array, int32, log
from numpy.core.fromnumeric import mean, std
from numpy.lib.scimath import sqrt
from pymc import MCMC
from thesis.scripts.bayesian.poisson import sme_calc_nocl, mape_calc, pred_calc, \
    rsqr_calc, poisson_req, poisson_req_nocl
from thesis.scripts.clustering.kmeans import kmeans
from thesis.scripts.dataset.dataset import weeklydataset
from thesis.scripts.hmm.hmm import HMM
from thesis.scripts.samples.aggregatesamples import aggregatebymins
from thesis.scripts.svr.svr import SVR
from time import time
import rvm_binding
from scipy.stats import norm
from rvm import VectorSample, RegressionTrainer, RadialBasisKernel, PolynomialKernel
import csv
import operator

def divide_by_cmd(filename1, filename2, position):
    X, label = weeklydataset(filename1, [])
    X2, label2 = weeklydataset(filename2, [])
    
    x = X + X2
    
    f = operator.itemgetter(position)
    
    commands = map(f, x)
    
    print commands[0:20]
    
    unique_cmd_set = set(commands)
    
    print unique_cmd_set
    
    unique_cmd = []
    
    while len(unique_cmd_set) > 0:
        
        unique_cmd.append(unique_cmd_set.pop())
        
    print unique_cmd
    
    files = []
    
    for i in range(len(unique_cmd)):
        files.append(csv.writer(open(unique_cmd[i]+"_cmd.csv", "wb"), delimiter=";"))
        
    
    for elem in x:
        index = unique_cmd.index(elem[position])
        files[index].writerow(elem)
    

def clusterize():
    X, label = weeklydataset('/home/work/Workloads/WmProxyWL/train.csv', [])
    X2, label2 = weeklydataset('/home/claudio/Workloads/WmProxyWL/test.csv', [])
    
    x = X + X2
    start_time = time()
    centroids, clusters = kmeans(x, 4)
    end_time = time()
    
    print end_time - start_time
    
    print centroids
    print len(clusters[0])
    print len(clusters[1])
    print len(clusters[2])
    print len(clusters[3])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster0.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 0", len(clusters[0])])
    results.writerows(clusters[0])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster1.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 1", len(clusters[1])])
    results.writerows(clusters[1])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster2.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 2", len(clusters[2])])
    results.writerows(clusters[2])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster3.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 3", len(clusters[3])])
    results.writerows(clusters[3])
    
def train_test(data, date_test):
    
    train = []
    test = []
    
    f = operator.itemgetter(0)
    
    timestamps = map(f, data)
    
    float_stamps = []
    
    for item in timestamps[1:len(timestamps)]:
        float_item = float(item)
        
        #shift to start from monday
        if (float_item > date_test):
            test.append(float_item - 131348.0)
        else:
            train.append(float_item - 131348.0)
    
    return train, test
    
    
def initialize_wmproxy():
    
    #get all the logs divided by commands (assumption)
    wmpcommon = csv.reader(open("/home/claudio/GenericWorkloadModeler/workloads/WMproxy/wmpcommon_cmd.csv"), delimiter = ';')
#    wmpcoreoperation = csv.reader(open("/home/work/Workloads/WmProxyWL/wmpcoreoperation.csv"), delimiter = ';')
#    wmp2wm = csv.reader(open("/home/work/Workloads/WmProxyWL/wmp2wm.csv"), delimiter = ';')
#    WMPAuthorizer = csv.reader(open("/home/work/Workloads/WmProxyWL/WMPAuthorizer.csv"), delimiter = ';')
#    WMPEventlogger = csv.reader(open("/home/work/Workloads/WmProxyWL/WMPEventlogger.csv"), delimiter = ';')
#    wmpoperations = csv.reader(open("/home/work/Workloads/WmProxyWL/wmpoperations.csv"), delimiter = ';')
#    wmpproxy = csv.reader(open("/home/work/Workloads/WmProxyWL/wmpproxy.csv"), delimiter = ';')
#    wmputils = csv.reader(open("/home/work/Workloads/WmProxyWL/wmputils.csv"), delimiter = ';')
    
    cluster0 = []
#    cluster1 = []
#    cluster2 = []
#    cluster3 = []
    
    for item in wmpcommon:
        cluster0.append(item)
    
#    for item in c1:
#        cluster1.append(item)    
#    
#    for item in c2:
#        cluster2.append(item)
#    
#    for item in c3:
#        cluster3.append(item)
    
    #for the moment I'll take just the first cluster...
    #devide train and test data: test data are from the next second of the next week
    train, test = train_test(cluster0, 1319545748.0) 
    
    
    
    #get the first column of the datapoints (timestamp) and aggregate it (both train and test)
    traininput, traintarget = aggregatebymins(train)
    testinput, testtarget = aggregatebymins(test)
    
    #set the testinput as the next week
    for i in range(len(testinput)):
        testinput[i] += 10080
    
    return traininput, traintarget, testinput, testtarget
    
    
    
def svr():
    
    #initialization of data wmproxy
    traininput, traintarget, testinput, testtarget = initialize_wmproxy()
    #training of the SVR
    
    #scaling values in training and test targets
    
    for i in range(len(traintarget)):
        if(traintarget[i] != 0):
            traintarget[i] = log(traintarget[i])
        if(traininput[i] != 0):
            traininput[i] = log(traininput[i])
            
    
    for i in range(len(testtarget)):
        if(testtarget[i] != 0):
            testtarget[i] = log(testtarget[i])
        if(testinput[i] != 0):
            testinput[i] = log(testinput[i])
    
    avg = mean(traintarget)
    sigma = std(traintarget)
    maxtrain = len(traintarget)
    C = max([abs(avg + sigma), abs(avg - sigma)])
    print "C is equal to %f" % C
    svr = SVR(traininput[maxtrain-1440:maxtrain], testinput, traintarget[maxtrain-1440:maxtrain],2**(-6),2**32,0.2,2**(-8))
    
    
    out = svr.svr_req(testinput[0:30])
    
    error = 0
    for i in range(len(out)):
        error += (out[i] - testtarget[i])
    
    mean_error = error / len(out)
    variance = 0
    for i in range(len(out)):
        variance = abs(out[i] - mean_error)
    
    variance /= len(out)
    
    print "Variance = %f" % variance
    
    epsilon = 3*variance*sqrt(log(len(out))/len(out))
    
    print "Epsilon = %f" % epsilon
    #calculation of the metrics
    sme = svr.calc_sme(testtarget[0:30], out)
    mape = svr.calc_mape(out, testtarget[0:30])
    predx = svr.calc_pred(out, testtarget[0:30], 25)
    rsq = svr.calc_rsqr(out, testtarget[0:30])
    print out
    print testtarget[0:30]
    # print model results!
    x = array(testinput[0:30], dtype=int32)
    y = array(testtarget[0:30], dtype=int32)
    xp = array(testinput[0:30], dtype=int32)
    yp = array(out, dtype=int32)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, y)
    ax1.plot(xp,yp,"r")
    ax1.axis([8.9,max(xp)+0.5,0,max(y)+10])
    ax1.set_xlabel('minutes of the week')
    ax1.set_ylabel('number of requests')
    fig.savefig("svr_model_%f" % time(), format='png')
    
    print "SME = %f" % sme
    print "MAPE = %f" % mape
    print "R^2 = %f" % rsq
    print "PREDX = %f" % predx
    

def hmm(states_nuber):
    
    #initialization of data wmproxy
    traininput, traintarget, testinput, testtarget = initialize_wmproxy()
    trainelements = len(traintarget)
#    models = HMM(traintarget, testinput, testtarget, 128, 128, max(traintarget+testtarget))
    model = HMM(traintarget, states_nuber)
#    vs = [hmm_req(models[j], target[j], testinput[j], testtarget[j], max(target[j])) for j in range(len(target)-1)]

##    model = hmm(target, testinput, testtarget, 6, 6, max(target))
#    v = models.hmm_req(traintarget, testinput[0:20], testtarget[0:20], max(traintarget))
#    counter = 0
#    for i in range(len(testtarget)):
#            if(testtarget[i] != 0):
#                testtarget[i] = numpy.log(testtarget[i])
#            else:
#                testtarget[i] = 1.0/100000000000
#
    states = model.hmm_req(testtarget[0:11], 30)
#    for v in vs:
#    lastest_states = [v[i][0][len(v[i][0])-1] for i in range(len(v)-1)]
#    print lastest_states
    
#    states = models.hmm_req(testtarget[0:10], 20)
#    
    ttarget = []
    
    print "States2"
    print states
    for state in states:
        li = model.m.getEmission(state)
        normal = norm.rvs(loc = li[0], scale = li[1], size=15)
        ttarget.append(normal)
#        counter += 1
##    sme = sme_calc(ttarget, testtarget[counter])
    print ttarget
    
    minout = []
    maxout = []
    meanout = []
    
    for element in ttarget:
        minout.append(min(element))
        maxout.append(max(element))
        meanout.append(mean(element))
        
    
#    print "minout %s: " % minout
#    print "meanout %s: " % meanout
#    print "maxout %s: " % maxout
     
    
    x = array(testinput[0:30], dtype=int32)
    y = array(testtarget[0:30], dtype=int32)
    xp = array(testinput[0:30], dtype=int32)
    yp = array(minout, dtype=int32)
    xp1 = array(testinput[0:30], dtype=int32)
    yp1 = array(maxout, dtype=int32)
    xp2 = array(testinput[0:30], dtype=int32)
    yp2 = array(meanout, dtype=int32)
    fig = figure()
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, y)
    ax1.plot(xp,yp,"r")
    ax1.plot(xp1,yp1,"g")
    ax1.plot(xp2,yp2,"y")
#    ax1.axis([8.9,max(xp)+0.5,0,max(y)+10])
    ax1.set_xlabel('minutes of the week')
    ax1.set_ylabel('number of requests')
    fig.savefig("hmm_model_%f.png" % time(), format='png')
    
#    sme = model.sme_calc(ttarget, testtarget[10:30])
#    mape = model.mape_calc(ttarget, testtarget[10:30])
#    predx = model.pred_calc(ttarget, testtarget[10:30], 25)
#    rsq = model.rsqr_calc(ttarget, testtarget[10:30])
#    
#    print "SME = %f" % sme
#    print "MAPE = %f" % mape
#    print "R^2 = %f" % rsq
#    print "PREDX = %f" % predx
    
    return model
    
def mcmc():
    
    
    from thesis.scripts.bayesian import requestModel_nocl
    model = MCMC(requestModel_nocl)
    
    traintarget = model.traintarget
    testtarget = model.testtarget
    traininput = model.traininput
    testinput = model.testinput
    
    
    starttime = time()
    model.sample(iter=1000, burn=200, thin=10)
    print "Training time"
    print time() - starttime
    
    for i in range(len(testinput)):
        testinput[i] -= 10080
    
    reqs = poisson_req_nocl(model, testinput[0:30], testtarget[0:30])
    
    ttarget = []
    for prob in reqs:
        
        m = max(prob)
        el = pylab.find(prob > m*2/3)
        maxes = nlargest(15, prob)
        maxvals = [prob.index(maxval) for maxval in maxes]
        ttarget.append(maxvals)
        
#    sme = sme_calc_nocl(ttarget, testtarget[0:20])
#    mape = mape_calc(ttarget, testtarget[0:20])
#    predx = pred_calc(ttarget, testtarget[0:20], 0.25)
#    rsq = rsqr_calc(ttarget, testtarget[0:20])    
#        
#    print "SME = %f" % sme
#    print "MAPE = %f" % mape
#    print "R^2 = %f" % rsq
#    print "PREDX = %f" % predx

    minout = []
    maxout = []
    meanout = []
    
    for element in ttarget:
        minout.append(min(element))
        maxout.append(max(element))
        meanout.append(mean(element))
        
    
    print "minout %s: " % minout
    print "meanout %s: " % meanout
    print "maxout %s: " % maxout
     
    
    x = array(testinput[0:30], dtype=int32)
    y = array(testtarget[0:30], dtype=int32)
    xp = array(testinput[0:30], dtype=int32)
    yp = array(minout, dtype=int32)
    xp1 = array(testinput[0:30], dtype=int32)
    yp1 = array(maxout, dtype=int32)
    xp2 = array(testinput[0:30], dtype=int32)
    yp2 = array(meanout, dtype=int32)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, y)
    ax1.plot(xp,yp,"r")
    ax1.plot(xp1,yp1,"g")
    ax1.plot(xp2,yp2,"y")
#    ax1.axis([8.9,max(xp)+0.5,0,max(y)+10])
    ax1.set_xlabel('minutes of the week')
    ax1.set_ylabel('number of requests')
    fig.savefig("mcmc_model_%f" % time(), format='png')

def rvr():
    samples = []
    
    #initialization of data wmproxy
    traininput, traintarget, testinput, testtarget = initialize_wmproxy()
    
    for x in traininput:
        samples.append((x,))
    starttime = time()
    smp = VectorSample(samples)
    msd = rvm_binding.compute_mean_squared_distance(smp)
    
    print msd
    
    gamma = 2**-6
    trainer = RegressionTrainer(RadialBasisKernel(gamma), 0.001)
#    trainer = RegressionTrainer(PolynomialKernel(gamma, 0, 5), 0.001)
    endtime = time()
    print 'using gamma of %f, calculated in %f' % (gamma, endtime - starttime)
    
    starttime = time()
    fn = trainer.train(samples, traintarget)
    endtime = time()
    
    print "Training time = ", (endtime - starttime)
    
    results = []
    shorttest = testinput[0:30]
    starttime = time()
    for test in shorttest:
        results.append(fn((test, )))
    
    endtime = time()
    
    print "Query time = ", (endtime - starttime)
    
    print results
    print testtarget[0:30]
    
    x = array(testinput[0:30], dtype=int32)
    y = array(testtarget[0:30], dtype=int32)
    xp = array(testinput[0:30], dtype=int32)
    yp = array(results, dtype=int32)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x, y)
    ax1.plot(xp,yp,"r")
    ax1.axis([10080,max(xp)+10,0,max(y)+10])
    ax1.set_xlabel('minutes of the week')
    ax1.set_ylabel('number of requests')
    fig.savefig("svr_model_%f" % time(), format='png')

if __name__ == '__main__':
    svr()
