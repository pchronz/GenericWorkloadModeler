'''
Created on Jul 12, 2011

@author: Claudio
'''
from shogun.Features import Labels, RealFeatures
from shogun.Kernel import GaussianKernel, PolyKernel
from shogun.Regression import LibSVR
from numpy import matrix, float64, array, int32
import time

class SVR:
    
    svr = None
    kernel = None
    feats_train = None
    
    def __init__(self, traininput,testinput,traintarget,width=0.5,C=1,epsilon=0.1,tube_epsilon=0.1):
    
        train = matrix(traininput, dtype=float64)
        test = matrix(testinput, dtype=float64)
        label_train = array(traintarget, dtype=float64)
        
        self.feats_train=RealFeatures(train)
        feats_test=RealFeatures(test)
        
        trainstart = time.time()
        self.kernel=GaussianKernel(self.feats_train, self.feats_train, width)
#        self.kernel = PolyKernel(self.feats_train, self.feats_train, 2, False)
        labels=Labels(label_train)
    
        self.svr=LibSVR(C, epsilon, self.kernel, labels)
        self.svr.set_tube_epsilon(tube_epsilon)
        self.svr.train()
        trainend = time.time()
        
        
        
        
        print 'SVR train time'
        print trainend-trainstart

    def svr_req(self,inputs):
           
        feat_inputs = RealFeatures(matrix(inputs, dtype=float64))
        
        teststart = time.time()
        self.kernel.init(self.feats_train, feat_inputs)   
        out = self.svr.classify(feat_inputs).get_labels()
            
#        feat_input0 = RealFeatures(matrix(inputs[0], dtype=float64))
#        feat_input1 = RealFeatures(matrix(inputs[1], dtype=float64))
#        feat_input2 = RealFeatures(matrix(inputs[2], dtype=float64))
#        feat_input3 = RealFeatures(matrix(inputs[3], dtype=float64))
#        
#        out.append(self.svr.classify(feat_input0).get_labels())
#        out.append(self.svr.classify(feat_input1).get_labels())
#        out.append(self.svr.classify(feat_input2).get_labels())
#        out.append(self.svr.classify(feat_input3).get_labels())
        testend = time.time()
        
        print 'SVR query response '
        print testend-teststart
        
        return out
    
    def calc_sme(self, testtarget, realtarget):
        result = 0.0
        for i in range(len(testtarget)):
            result += pow((realtarget[i] - testtarget[i]),2)
        result /= len(testtarget)
        
        return result
    
    def calc_mape(self, testtarget, realtarget):
        result = 0.0
        
        for i in range(len(testtarget)):
            result = abs(testtarget[i] - realtarget[i])/testtarget[i]
        
        return result/len(testtarget)
    
    def calc_rsqr(self, testtarget, realtarget):
        result_up = 0.0
        result_down = 0.0
        avg = sum(realtarget)/len(realtarget)
        
        for i in range(len(testtarget)):
            result_up += pow((realtarget[i] - testtarget[i]),2)
            result_down += (realtarget[i] - avg)
        
        return 1 - (result_up / result_down)
    
    def calc_pred(self, testtarget, realtarget, x):
        
        countx = 0.0
        
        for i in range(len(testtarget)):
            if ((testtarget[i]/realtarget[i]) - 1 < (realtarget[i] * (1-x))):
                countx += 1
        return countx / len(realtarget)