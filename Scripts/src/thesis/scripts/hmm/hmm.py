'''
Created on Sep 26, 2011

@author: work
'''

from numpy import array, int32, mean, std, zeros
import numpy
from collections import Counter
from ghmm import *
import types
import time
from sets import Set
from numpy.core.numeric import ones

class HMM():
    '''
    classdocs
    '''
    m = None
    sigma = None
    
    '''
    This constructor is used in case of continuous distribution for emissions
    '''
#    def __init__(self, traintarget, N):
#        self.sigma = Float()
#        A = []
#        B = []
#        
##        for i in range(len(traintarget)):
##            if(traintarget[i] != 0):
##                traintarget[i] = numpy.log(traintarget[i])
##            else:
##                traintarget[i] = 1.0/100000000000
#        
#        
##        for i in range(N):
##            transition = []
##            for j in range(N):
##                if(i == j):
##                    transition.append(1.0/10000000)
##                else:
##                    transition.append(1.0/(N-1))
##            A.append(transition)     
##        max+=1
##        targetlist = list(set(sortedtrain))
##        numbers_zero = sortedtrain.count(0)
##        print numbers_zero
##        targetlist = sortedtrain[(numbers_zero+(numbers_zero/3)):len(sortedtrain)]
#        
#        
#        ## Split the array in equal parts, in order to get the same number of points for each state
##        sortedtrain = sorted(traintarget)
##        times = len(sortedtrain)/N
##        print times
##        chunk = lambda ulist, step:  map(lambda i: ulist[i:i+step],  xrange(0, len(ulist), step))
##        
##        tempB = chunk(sortedtrain, times)
#
#        
#        ## Split values by time
#        times = len(traintarget)/N
#        chunk = lambda ulist, step:  map(lambda i: ulist[i:i+step],  xrange(0, len(ulist), step))
#        
#        tempB = chunk(traintarget, times)        
##        maxvalue = max(sortedtrain)
##        unit = int(maxvalue/N)
##        
##        tempB = []
##        print "Maxvalue = %d" % maxvalue
##        print "Unit is %d" % unit
##        for i in range(N):
##            tempB.append([])
##            
##        for value in sortedtrain:
##            if value/unit >= N:
##                tempB[N-1].append(value)
##            else:
##                tempB[value/unit].append(value)
#   
#        
##        tempB = traintarget[0:(numbers_zero+(numbers_zero/3))]+tempB
#        for tb in tempB:
#            if len(tb) > 0:
#                meanB = mean(tb)
#                varB = std(tb)
#                if(varB == 0):
#                    varB = 0.01
#                
#                print "mean = %f  var = %f" % (meanB, varB)
#            B.append([meanB, varB])
#        
#        pi = [1.0/N]*N
#        
#        for i in range(len(B)):
#            transition = []
#            for j in range(len(B)):
#                if(i == j):
#                    transition.append(0)
#                else:
#                    transition.append(1.0/(len(B)-1))
#            A.append(transition)
#        
#        
#        print "B = %s" % B
#        
#        self.m = HMMFromMatrices(self.sigma, GaussianDistribution(self.sigma), A, B, pi)
#        trainstart = time.time()
#        train = EmissionSequence(self.sigma, traintarget)
#        self.m.baumWelch(train)
#        trainend = time.time()
#        print 'HMM train time'
#        print trainend - trainstart
#        
        
    '''
    This constructor is used in case of discrete distribution of emissions
    '''
    def __init__(self, fm_train, N, max):
        self.sigma = IntegerRange(1, max+1)
        
        A = []
        B = []
        
        #Fair transition matrix
        A = ones([N,N])/(N*1.0)
#        print A
        
        # Not fair transition matrix
#        for i in range(N):
#            transition = []
#            for j in range(N):
#                if(i == j):
#                    transition.append(1.0/10000000)
#                else:
#                    transition.append(1.0/(N-1))
#            A.append(transition)   
                
        # now data are divided in order to obtain a starting emission values for each state.
        # number of partition basing on the number of states
#        
#        npartition = int(len(fm_train)/N)
#        
#        
#        # for the first N-1 states
#        for i in range(N-1):
#            submatrix = fm_train[i*npartition:((1+i)*npartition)-1]
#            occurrences = Counter(submatrix)
#            emission = [0.0]*(max)
#            
#            for key in occurrences.keys():
#                emission[key-1] = occurrences[key-1]
#            
#            
#            emission = array(emission) / (sum(emission))
#            
#            # The matrix is too sparse so an adjustement is needed: all value equal to 0 will be set to 0.0001
#            # and the normal value will be adjusted in order to have th e sum of emission equal to 1!
#            
#            numberofzeros = list(emission).count(0)
#            adjvalue = (numberofzeros * 0.0001) / (emission != 0).sum()
#            
#            for i in range(0,len(emission)):
#                if emission[i] == 0:
#                    emission[i] = 0.0001
#                else:
#                    emission[i] -= adjvalue
#            
#            
#            
#            B.append(list(emission))
#        
#        # the last state is computed apart because it can have different number of values!
#        submatrix = fm_train[N-1*npartition:len(fm_train)-1]
#        occurrences = Counter(submatrix)
#        emission = [0.0]*(max)
#        
#        print occurrences
#        
#        for key in occurrences.keys():
#            emission[key-1] = occurrences[key-1]
#        
#        
#        emission = array(emission) / (sum(emission))
#        
#        # The matrix is too sparse so an adjustement is needed: all value equal to 0 will be set to 0.0001
#        # and the normal value will be adjusted in order to have th e sum of emission equal to 1!
#        
#        numberofzeros = list(emission).count(0)
#        adjvalue = numberofzeros * 0.0001 / (emission != 0).sum()
#        
#        for i in range(0,len(emission)):
#            if emission[i] == 0:
#                emission[i] = 0.0001
#            else:
#                emission[i] -= adjvalue
#        
#        B.append(list(emission))


        # The emission distribution is computed basing on the output range, each state has singl range value where the probability
        # is equal and the other values have 0.0001 prob percentage
        B = []
        partition = int(max/N)
        
        # get occurrences of each value
        occurrences_count = Counter(fm_train)
        occurrences_prob = zeros(264)
        
        for i in occurrences_count.keys():
            occurrences_prob[i] = occurrences_count[i]
        
        
        for i in range(N):
            emission = ones(max) * 0.0001
            emission[i*partition:(i+1)*partition] = (occurrences_prob[i*partition:(i+1)*partition]/sum(occurrences_prob[i*partition:(i+1)*partition])) - (((max - partition)*0.0001)/partition)
            print sum(emission)
            B.append(emission)
        
        # Adjusting the NaN values of the emission matrix
        for i in range(N):
            if(any(numpy.isnan(B[i]))):
                B[i]= list(ones(max) / max)
                
        
         
        print "B = %s" % B
        
        pi = [1.0/N]*N
        self.m = HMMFromMatrices(self.sigma, DiscreteDistribution(self.sigma), A, list(B), pi)
        train = EmissionSequence(self.sigma, fm_train)
        trainstart = time.time()
        self.m.baumWelch(train)
        trainend = time.time()
        print 'HMM train time'
        print trainend - trainstart
        
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
        print(starttest)
        states = []
        seq = EmissionSequence(self.sigma, starttest)
        
        for i in range(timewindow):
            viterbipath = self.m.viterbi(seq)
            
            print "viterbi"
            print viterbipath
            
            A = self.m.asMatrices()[0]
            states = list(viterbipath[0])
            laststate = states[len(states)-1]
            ind = int(laststate)
            
            states.append(A[ind].index(max(A[ind])))
            ind = int(states[len(states)-1])
            
            laststateemission = self.m.getEmission(A[ind].index(max(A[ind])))
            bestvalue = laststateemission.index(max(laststateemission))
            print bestvalue
            starttest.append(bestvalue)
            
            newtest = starttest[1:len(starttest)]
            
            
            seq = EmissionSequence(self.sigma, newtest)
        
        testend = time.time()
        print "HMM query response"
        print testend-teststart
        predictedstates = states[len(states)-timewindow:len(states)]
        print "Predicted States"
        print predictedstates
        return predictedstates
        
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