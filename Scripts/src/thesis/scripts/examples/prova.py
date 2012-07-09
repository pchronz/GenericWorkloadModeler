'''
Created on Jun 26, 2012

@author: work
'''

from thesis.scripts.samples.aggregatesamples import aggregatebymins
import csv
import operator
import numpy

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
    wmpcommon = csv.reader(open("/media/DATA/Thesis/Workloads/GenericWorkloadModeler/workloads/WMproxy/wmpcommon_cmd.csv"), delimiter = ';')
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
    #traintarget, testarget scaling
    
    for i in range(len(traintarget)):
        if(traintarget[i] != 0):
            traintarget[i] = numpy.log(traintarget[i])
        else:
            traintarget[i] = 1.0/100000000000
            
            
    for i in range(len(testtarget)):
        if(testtarget[i] != 0):
            testtarget[i] = numpy.log(testtarget[i])
        else:
            testtarget[i] = 1.0/100000000000    
    libtrain = csv.writer(open("scaledtrain.csv", "wb"), delimiter= " ")
    libtest = csv.writer(open("scaledtest.csv", "wb"), delimiter= " ")
    
    for i in range(len(traintarget)):
        libtrain.writerow([traintarget[i], traininput[i]])
        
    for i in range(30):
        libtest.writerow([testtarget[i], testinput[i]])
    
    return traininput, traintarget, testinput, testtarget