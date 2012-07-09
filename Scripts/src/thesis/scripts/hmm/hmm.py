'''
Created on Sep 26, 2011

@author: work
'''

from numpy import array, int32, mean, std, zeros
import numpy
from ghmm import *
import types
import time
from sets import Set

class HMM():
    '''
    classdocs
    '''
    m = None
    sigma = None
    
    def __init__(self, traintarget, N):
        self.sigma = Float()
        A = []
        B = []
        
#        for i in range(len(traintarget)):
#            if(traintarget[i] != 0):
#                traintarget[i] = numpy.log(traintarget[i])
#            else:
#                traintarget[i] = 1.0/100000000000
        
        
#        for i in range(N):
#            transition = []
#            for j in range(N):
#                if(i == j):
#                    transition.append(1.0/10000000)
#                else:
#                    transition.append(1.0/(N-1))
#            A.append(transition)     
#        max+=1
        sortedtrain = sorted(traintarget)
#        targetlist = list(set(sortedtrain))
#        numbers_zero = sortedtrain.count(0)
#        print numbers_zero
#        targetlist = sortedtrain[(numbers_zero+(numbers_zero/3)):len(sortedtrain)]
        
#        times = len(sortedtrain)/N
#        print times
#        chunk = lambda ulist, step:  map(lambda i: ulist[i:i+step],  xrange(0, len(ulist), step))
#        
#        tempB = chunk(targetlist, times)
        
        maxvalue = max(sortedtrain)
        unit = int(maxvalue/N)
        
        tempB = []
        print "Maxvalue = %d" % maxvalue
        print "Unit is %d" % unit
        for i in range(N):
            tempB.append([])
            
        for value in sortedtrain:
            if value/unit >= N:
                tempB[N-1].append(value)
            else:
                tempB[value/unit].append(value)
   
        
#        tempB = traintarget[0:(numbers_zero+(numbers_zero/3))]+tempB
        for tb in tempB:
            if len(tb) > 0:
                meanB = mean(tb)
                varB = std(tb)
                if(varB == 0):
                    varB = 1.0
                
                print "mean = %f  var = %f" % (meanB, varB)
            B.append([meanB, varB])
        
        pi = [1.0/N]*N
        
        for i in range(len(B)):
            transition = []
            for j in range(len(B)):
                if(i == j):
                    transition.append(0)
                else:
                    transition.append(1.0/(len(B)-1))
            A.append(transition)
        
        
        print "B = %s" % B
        
        self.m = HMMFromMatrices(self.sigma, GaussianDistribution(self.sigma), A, B, pi)
        trainstart = time.time()
        train = EmissionSequence(self.sigma, traintarget)
        self.m.baumWelch(train)
        trainend = time.time()
        print 'HMM train time'
        print trainend - trainstart
#    def __init__(self, fm_train, testinput, testtarget, N, M, max):
#        self.sigma = IntegerRange(0, max+1)
#        
#        A = []
#        B = []
#        print max
#        for i in range(N):
#            B.append([0.00001]*(max+1))
#            emission = []
#            for i in range(N):
#                emission.append(1.0/N)
#            A.append(emission)
#            
#        partition = max/N        
##        max+=1
#        
#        disc = 1 - 0.00001 * (max - partition)
#        
#        for i in range(len(B)):
#            for j in range(max):
#                if(j >= i*partition) and (j < (i+1)*partition):
#                    B[i][j] = (disc/partition)
#                    
#        pi = [1.0/N]*N
#        self.m = HMMFromMatrices(self.sigma, DiscreteDistribution(self.sigma), A, B, pi)
#        train = EmissionSequence(self.sigma, fm_train)
#        trainstart = time.time()
#        self.m.baumWelch(train)
#        trainend = time.time()
#        print 'HMM train time'
#        print trainend - trainstart
        
        #delete silent states
    
#    def __init__(self, fm_train, testinput, testtarget, N, M, max):
#        '''
#        Constructor
#        '''
#        sigma = IntegerRange(0, max+1)
#        A = [[1.0/6]*6]*6
#    #    efairnum = max+1
#    #    efair = [1.0/efairnum]*efairnum
#        S0 = []
#        partition = max/6.0
#        max+=1
#        disc = 1 - (0.00001 * (max-partition))
#    #    print disc
#        for i in range(max):
#            if i < partition:
#                S0.append(disc/partition)
#            else:
#                S0.append(0.00001)
#        S1 = []
#        for i in range(max):
#            if (i >= partition) and (i < 2*partition):
#                S1.append(disc/partition)
#            else:
#                S1.append(0.00001)
#        S2 = []
#        for i in range(max):
#            if (i >= 2*partition) and (i < 3*partition):
#                S2.append(1/max)
#            else:
#                S2.append(0.00001)
#        S3 = []
#        for i in range(max):
#            if (i >= 3*partition) and (i < 4*partition):
#                S3.append(disc/partition)
#            else:
#                S3.append(0.00001)
#        S4 = []
#        for i in range(max):
#            if (i >= 4*partition) and (i < 5*partition):
#                S4.append(disc/partition)
#            else:
#                S4.append(0.00001)
#        S5 = []
#        for i in range(max):
#            if (i >= 5*partition):
#                S5.append(disc/partition)
#            else:
#                S5.append(0.00001)
#    
#        B  = [S0,S1,S2,S3,S4,S5]
#        pi = [1.0/6] * 6
#        self.m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
#        train = EmissionSequence(sigma, fm_train)
#        trainstart = time.time()
#        self.m.baumWelch(train)
#        trainend = time.time()
#        print 'HMM train time'
#        print trainend - trainstart

#    def hmm_req(self, fm_train, testinput, testtarget, max):
#        sigma = IntegerRange(0, max+1)
#        v = []
#        
#        teststart = time.time()
#        for t in testinput:
#            seq = fm_train[0:t]
#            seq_test = EmissionSequence(sigma, seq)
#            v.append(self.m.viterbi(seq_test))
#        testend = time.time()
#        
#        print "HMM query response"
#        print testend-teststart
#        return v
    def hmm_req(self, starttest, timewindow):
        teststart = time.time()
        seq = EmissionSequence(self.sigma, starttest)
        viterbipath = self.m.viterbi(seq)
        
        A = self.m.asMatrices()[0]
        states = list(viterbipath[0])
        print "States"
        print states
        laststate = states[len(states)-1]
        ind = int(laststate)
        print ind
        print A[ind].index(max(A[ind]))
        
        for i in range(timewindow - len(starttest)):
            states.append(A[ind].index(max(A[ind])))
            ind = int(states[len(states)-1])
        
        testend = time.time()
        print "HMM query response"
        print testend-teststart
        return states
        
    def sme_calc(self, testtarget, realtarget):
        result = 0.0
        for i in range(len(testtarget)):
            dis = min([pow(realtarget[i] - testtarget[i][j], 2) for j in range(len(testtarget[i]))])
            result += dis
        return result/len(testtarget[0])
    
    def mape_calc(self, testtarget, realtarget):
        result = 0.0
        
        for i in range(len(testtarget)):
            result += min([abs(testtarget[i][j] + 1 - realtarget[i])/(testtarget[i][j] + 1) for j in range(len(testtarget[i]))])
        
        return result/len(testtarget)
    
    def rsqr_calc(self, testtarget, realtarget):
        result_up = 0.0
        result_down = 0.0
        avg = sum(realtarget)/len(realtarget)
        
        for i in range(len(testtarget)):
            result_up += min([pow((realtarget[i] - testtarget[i][j]),2) for j in range(len(testtarget[i]))])
            result_down += (realtarget[i] - avg)
        
        return 1 - (result_up / result_down)
    
    def pred_calc(self, testtarget, realtarget, x):
        countx = 0.0
        for i in range(len(testtarget)):
            min_error = min([(testtarget[i][j]/realtarget[i]) - 1 for j in range(len(testtarget[i]))])
            if (min_error < (realtarget[i] * (1-x))):
                countx += 1
        return countx / len(realtarget)