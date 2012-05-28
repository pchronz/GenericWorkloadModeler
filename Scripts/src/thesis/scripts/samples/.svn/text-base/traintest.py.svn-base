'''
Created on Jul 19, 2011

@author: work
'''
from random import randint

def traintest(input, target, testP, transpose):
    
    if transpose == 1:
        testValue  = ((len(input))*testP)/100
        testLines = list()
        partition = len(input)/4
        
        for i in range(testValue/4):
            randomNumber = randint(1,partition)
            while (randomNumber in testLines) is True:
                randomNumber = randint(1,partition)
            testLines.append(randomNumber)
        
        for j in range(testValue/4):
            randomNumber = randint(partition+1,2*partition)
            while (randomNumber in testLines) is True:
                randomNumber = randint(partition+1,2*partition)
            testLines.append(randomNumber)
        
        for k in range(testValue/4):
            randomNumber = randint(2*partition+1,3*partition)
            while (randomNumber in testLines) is True:
                randomNumber = randint(2*partition+1,3*partition)
            testLines.append(randomNumber)
        
        for l in range(testValue/4):
            randomNumber = randint(3*partition+1,len(input))
            while (randomNumber in testLines) is True:
                randomNumber = randint(3*partition+1,len(input))
            testLines.append(randomNumber)
        traininput = list()
        traintarget = list()
        testinput = list()
        testtarget = list()
        testLines.sort()
        countTest = 0
        for counter in range(len(input)):
            if countTest < len(testLines) and counter == testLines[countTest]:
                testinput.append(input[counter])
                testtarget.append(target[counter])
                countTest +=1
            else:
                traininput.append(input[counter])
                traintarget.append(target[counter])
        return traininput, traintarget, testinput, testtarget