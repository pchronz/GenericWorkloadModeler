'''
Created on Sep 20, 2011

@author: work
'''
from collections import Counter
from numpy import array, int32, absolute
from pymc import *
from time import time
import math

def poisson_req(model, testinput, cluster):
    
    requests_prob = []
    starttime = time()
    for input in testinput:
        rate = model.rates[cluster][input].trace()
        rate_array = array(rate, dtype = int32)
        count_rate = Counter(rate_array)
        highest_rate = count_rate.most_common()[0][0]
        
        req = [math.exp((poisson_like(x, highest_rate))) for x in range(200)]
        requests_prob.append(req)
    print "Query time"
    print time() - starttime 
    return requests_prob
        
def poisson_req_nocl(model, testinput, testtarget):
    
    requests_prob = []
    starttime = time()
    for input in testinput:
        rate = model.rates[input].trace()
        rate_array = array(rate, dtype = int32)
        count_rate = Counter(rate_array)
        highest_rate = count_rate.most_common()[0][0]
        
        req = [math.exp((poisson_like(x, highest_rate))) for x in range(max(testtarget))]
        requests_prob.append(req)
    print "Query time"
    print time() - starttime 
    return requests_prob
       
def sme_calc(testtarget,  model, cluster):
    result = 0.0
    for i in range(len(testtarget)):
        meanrealreq = (sum(model.target[cluster][i])/len(model.target[cluster][i]))
        dis = min([pow( meanrealreq - testtarget[i][j], 2) for j in range(len(testtarget[i]))])
        result += dis
    return result/len(testtarget)

def sme_calc_nocl(testtarget, realtarget):
    result = 0.0
    for i in range(len(testtarget)):
        print testtarget[i]
        print realtarget[i]
        dis = min([pow(realtarget[i] - testtarget[i][j], 2) for j in range(len(testtarget[i]))])
        result += dis
    return result/len(testtarget)
    
def mape_calc(testtarget, realtarget):
    result = 0.0
    print realtarget
    for i in range(len(testtarget)):
	print testtarget[i]
        result += min([absolute(float(testtarget[i][j]) - realtarget[i])/testtarget[i][j] for j in range(len(testtarget[i]))])
    
    return result/len(testtarget)

def rsqr_calc(testtarget, realtarget):
    result_up = 0.0
    result_down = 0.0
    avg = sum(realtarget)/len(realtarget)
    
    for i in range(len(testtarget)):
        result_up += min([pow((realtarget[i] - testtarget[i][j]),2) for j in range(len(testtarget[i]))])
        result_down += pow((realtarget[i] - avg),2)
    
    return 1 - (result_up / result_down)

def pred_calc(testtarget, realtarget, x):
    countx = 0.0
    for i in range(len(testtarget)):
        min_error = min([(float(testtarget[i][j])/realtarget[i]) - 1 for j in range(len(testtarget[i]))])
        if (min_error < (float(realtarget[i]) * (1-x))):
            countx += 1
    return countx / len(realtarget)
