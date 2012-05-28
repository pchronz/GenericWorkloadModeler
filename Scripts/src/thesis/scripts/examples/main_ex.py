'''
Created on Jul 13, 2011

@author: work
'''
from heapq import nlargest
from numpy import matrix, array, savetxt, genfromtxt
import pylab
from thesis.scripts.clustering.clustering import create_clustered_samples,\
    create_clustered_samples_ndata
from thesis.scripts.dataset.dataset import weeklydataset_shogun, \
    weeklydataset_sg_ndata
from thesis.scripts.hmm.hmm import hmm, hmm_req, sme_calc
from thesis.scripts.samples.aggregatesamples import aggregateby10mins_sg_mean, \
    aggregateby10mins_sg_mcmc, aggregateby10mins_sg_ndata,\
    aggregateby10mins_sg_mcmc_ndata
from thesis.scripts.samples.traintest import traintest
import sys
from thesis.scripts.examples.svrmodel import svrmodel
from thesis.scripts.hmm.HMM import HMM
#from thesis.scripts.dataset.getdata_sg import load_cubes
def create_clusters():
    [points, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_mod.csv', [])
#    [points, label] = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/ewsdata2.csv', [])
#    print points
    clusteredpoints, cdata = create_clustered_samples(points, 10, 1)
#    clusteredpoints, cdata = create_clustered_samples_ndata(points, 3, 1)
    
    cluster0 = clusteredpoints[0][1]
    cluster1 = clusteredpoints[1][1]
    cluster2 = clusteredpoints[2][1]
    cluster3 = clusteredpoints[3][1]
    cluster4 = clusteredpoints[4][1]
    cluster5 = clusteredpoints[5][1]
    cluster6 = clusteredpoints[6][1]
    cluster7 = clusteredpoints[7][1]
    cluster8 = clusteredpoints[8][1]
    cluster9 = clusteredpoints[9][1]
    print type(cluster0)
    savetxt("cluster0.csv", matrix(clusteredpoints[0]), delimiter=';')
    savetxt("cluster1.csv", matrix(clusteredpoints[1]), delimiter=';')
    savetxt("cluster2.csv", matrix(clusteredpoints[2]), delimiter=';')
    savetxt("cluster3.csv", matrix(clusteredpoints[3]), delimiter=';')
    savetxt("cluster4.csv", matrix(clusteredpoints[4]), delimiter=';')
    savetxt("cluster5.csv", matrix(clusteredpoints[5]), delimiter=';')
    savetxt("cluster6.csv", matrix(clusteredpoints[6]), delimiter=';')
    savetxt("cluster7.csv", matrix(clusteredpoints[7]), delimiter=';')
    savetxt("cluster8.csv", matrix(clusteredpoints[8]), delimiter=';')
    savetxt("cluster9.csv", matrix(clusteredpoints[9]), delimiter=';')
#    
#    clusterlen = [len(cluster0[0]), len(cluster1[0]), len(cluster2[0]),len(cluster3[0]),len(cluster4[0]), len(cluster5[0]),
#                  len(cluster6[0]), len(cluster7[0]), len(cluster8[0]),len(cluster9[0])]
#    m = max(clusterlen)
#    minimum = min(clusterlen)
#    
#    maxcluster = clusterlen.index(m)
#    mincluster = clusterlen.index(minimum)
#    input, target = aggregateby10mins_sg(clusteredpoints[mincluster][0])
    print "Cluster0 points: %d" % len(cluster0)
    print "Cluster1 points: %d" % len(cluster1)
    print "Cluster2 points: %d" % len(cluster2)
    print "Cluster3 points: %d" % len(cluster3)
    print "Cluster4 points: %d" % len(cluster4)
    print "Cluster5 points: %d" % len(cluster5)
    print "Cluster6 points: %d" % len(cluster6)
    print "Cluster7 points: %d" % len(cluster7)
    print "Cluster8 points: %d" % len(cluster8)
    print "Cluster9 points: %d" % len(cluster9)
    
    
    
def main_hmm():
#    [points, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_mod.csv', [])
##    [points, label] = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/ewsdata2.csv', [])
##    print points
#    clusteredpoints, cdata = create_clustered_samples(points, 10, 1)
##    clusteredpoints, cdata = create_clustered_samples_ndata(points, 3, 1)
#    
    clusteredpoints = []
    clusteredpoints.append(genfromtxt("cluster0.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster1.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster2.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster3.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster4.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster5.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster6.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster7.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster8.csv", delimiter=';'))
    clusteredpoints.append(genfromtxt("cluster9.csv", delimiter=';'))
    
    cluster0 = clusteredpoints[0][1]
    cluster1 = clusteredpoints[1][1]
    cluster2 = clusteredpoints[2][1]
    cluster3 = clusteredpoints[3][1]
    cluster4 = clusteredpoints[4][1]
    cluster5 = clusteredpoints[5][1]
    cluster6 = clusteredpoints[6][1]
    cluster7 = clusteredpoints[7][1]
    cluster8 = clusteredpoints[8][1]
    cluster9 = clusteredpoints[9][1]
##    
##    clusterlen = [len(cluster0[0]), len(cluster1[0]), len(cluster2[0]),len(cluster3[0]),len(cluster4[0]), len(cluster5[0]),
##                  len(cluster6[0]), len(cluster7[0]), len(cluster8[0]),len(cluster9[0])]
##    m = max(clusterlen)
##    minimum = min(clusterlen)
##    
##    maxcluster = clusterlen.index(m)
##    mincluster = clusterlen.index(minimum)
##    input, target = aggregateby10mins_sg(clusteredpoints[mincluster][0])
    print "Cluster0 points: %d" % len(cluster0)
    print "Cluster1 points: %d" % len(cluster1)
    print "Cluster2 points: %d" % len(cluster2)
    print "Cluster3 points: %d" % len(cluster3)
    print "Cluster4 points: %d" % len(cluster4)
    print "Cluster5 points: %d" % len(cluster5)
    print "Cluster6 points: %d" % len(cluster6)
    print "Cluster7 points: %d" % len(cluster7)
    print "Cluster8 points: %d" % len(cluster8)
    print "Cluster9 points: %d" % len(cluster9)
    clusterlen = [len(cluster0), len(cluster1), len(cluster2), len(cluster3), len(cluster4), len(cluster5), len(cluster6), len(cluster7), len(cluster8), len(cluster9)]
    minimum = min(clusterlen)
    mincluster = clusterlen.index(70400)
    input = []
    target = []
    numcluster = 0
#    input, target = aggregateby10mins_sg_ndata(points[1], 0)
    for cluster in clusteredpoints:
        inp, tar = aggregateby10mins_sg_mean(cluster[0], numcluster)
#        inp, tar = aggregateby10mins_sg_ndata(cluster[1], numcluster)
        input.append(inp)
        target.append(tar)
        numcluster += 1
#    input, target = [aggregateby10mins_sg_ndata(cluster[0]) for cluster in clusteredpoints]
    traininput = []
    traintarget = []
    testinput = []
    testtarget = []
    for i in range(len(input)):
        trainin, traintar, testin, testtar = traintest(input[i], target[i], 20, 1)
        traininput.append(trainin)
        traintarget.append(traintar)
        testinput.append(testin)
        testtarget.append(testtar)
    
#    traininput, traintarget, testinput, testtarget = traintest(input, target, 20, 1)
#    models = [HMM(target[j], testinput[j], testtarget[j], 6, 6, max(target[j])) for j in range(len(target))]
    print "cluster maximum = %f" % max(target[mincluster])
    models = hmm(target[mincluster], testinput[mincluster], testtarget[mincluster], 6, 6, max(target[mincluster]))
#    vs = [hmm_req(models[j], target[j], testinput[j], testtarget[j], max(target[j])) for j in range(len(target)-1)]

##    model = hmm(target, testinput, testtarget, 6, 6, max(target))
    v = hmm_req(models, target[mincluster], testinput[mincluster], testtarget[mincluster], max(target[mincluster]))
#    counter = 0
#
#    for v in vs:
    lastest_states = [v[i][0][len(v[i][0])-1] for i in range(len(v)-1)]
    print lastest_states
    
    ttarget = []
    
    for state in lastest_states:
        li = models.getEmission(state)
        m = (max(li)* 2.0)/3.0
        el = pylab.find(array(li) > m)
        maxes = nlargest(10, li)
        maxvals = [li.index(maxval) for maxval in maxes]
        ttarget.append(maxvals)
#        counter += 1
##    sme = sme_calc(ttarget, testtarget[counter])
    sme = sme_calc(ttarget, testtarget[mincluster])
    print "SME = %f" % sme
#        counter += 1
    return models
    
def clusterize():
    [points, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_mod.csv', [])
#    points, label = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/ewsdata2.csv', [])
#    print points
    clusteredpoints, cdata = create_clustered_samples(points, 10, 1)
#    clusteredpoints, cdata = create_clustered_samples_ndata(points, 3, 1)
    cluster0 = clusteredpoints[0][1]
    cluster1 = clusteredpoints[1][1]
    cluster2 = clusteredpoints[2][1]
    cluster3 = clusteredpoints[3][1]
    cluster4 = clusteredpoints[4][1]
    cluster5 = clusteredpoints[5][1]
    cluster6 = clusteredpoints[6][1]
    cluster7 = clusteredpoints[7][1]
    cluster8 = clusteredpoints[8][1]
    cluster9 = clusteredpoints[9][1]
#    
#    clusterlen = [len(cluster0[0]), len(cluster1[0]), len(cluster2[0]),len(cluster3[0]),len(cluster4[0]), len(cluster5[0]),
#                  len(cluster6[0]), len(cluster7[0]), len(cluster8[0]),len(cluster9[0])]
#    m = max(clusterlen)
#    minimum = min(clusterlen)
#    
#    maxcluster = clusterlen.index(m)
#    mincluster = clusterlen.index(minimum)
#    input, target = aggregateby10mins_sg(clusteredpoints[mincluster][0])
    print "Cluster0 points: %d" % len(cluster0)
    print "Cluster1 points: %d" % len(cluster1)
    print "Cluster2 points: %d" % len(cluster2)
    print "Cluster3 points: %d" % len(cluster3)
    print "Cluster4 points: %d" % len(cluster4)
    print "Cluster5 points: %d" % len(cluster5)
    print "Cluster6 points: %d" % len(cluster6)
    print "Cluster7 points: %d" % len(cluster7)
    print "Cluster8 points: %d" % len(cluster8)
    print "Cluster9 points: %d" % len(cluster9)
    input = []
    target = []
    numcluster = 0
    for cluster in clusteredpoints:
        inp, tar = aggregateby10mins_sg_mean(cluster[0], numcluster)
#        inp, tar = aggregateby10mins_sg_ndata(cluster[1], numcluster)
        input.append(inp)
        target.append(tar)
        numcluster += 1
#    input, target = [aggregateby10mins_sg_ndata(cluster[0]) for cluster in clusteredpoints]
    traininput = []
    traintarget = []
    testinput = []
    testtarget = []
    for i in range(len(input)):
        trainin, traintar, testin, testtar = traintest(input[i], target[i], 20, 1)
        traininput.append(trainin)
        traintarget.append(traintar)
        testinput.append(testin)
        testtarget.append(testtar)

    return traininput, traintarget, testinput, testtarget, cdata
#    hmm(target, 6, 6, max(target))

def main_svr():
    
    traininput, traintarget, testinput, testtarget, cdata = clusterize()
    
    for i in range(len(traininput)):
        svrmodel(traininput[i], traintarget[i], testinput[i], testtarget[i])
    
def main_mcmc():
    [points, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_mod.csv', [])
#    points, label = weeklydataset_sg_ndata('/media/4AC0AB31C0AB21E5/Documents and Settings/Claudio/Documenti/Thesis/Workloads/MSClaudio/ews/ewsdata2.csv', [])
    clusteredpoints, cdata = create_clustered_samples(points, 10, 1)
#    clusteredpoints, cdata = create_clustered_samples_ndata(points, 3, 1)
#    cluster0 = clusteredpoints[0]
#    cluster1 = clusteredpoints[1]
#    cluster2 = clusteredpoints[2]
#    cluster3 = clusteredpoints[3]
#    cluster4 = clusteredpoints[4]
#    cluster5 = clusteredpoints[5]
#    cluster6 = clusteredpoints[0]
#    cluster7 = clusteredpoints[1]
#    cluster8 = clusteredpoints[2]
#    cluster9 = clusteredpoints[3]
#    
#    clusterlen = [len(cluster0[0]), len(cluster1[0]), len(cluster2[0]),len(cluster3[0]),len(cluster4[0]), len(cluster5[0]),
#                  len(cluster6[0]), len(cluster7[0]), len(cluster8[0]),len(cluster9[0])]
#    m = max(clusterlen)
#    minimum = min(clusterlen)
#    
#    maxcluster = clusterlen.index(m)
#    mincluster = clusterlen.index(minimum)
    target = []
    numcluster = 0
    for cluster in clusteredpoints:
        target.append(aggregateby10mins_sg_mcmc(cluster[0], numcluster))
#        target.append(aggregateby10mins_sg_mcmc_ndata(cluster[1], numcluster))
        numcluster += 1
    return target

if __name__ == '__main__':
    main_hmm()