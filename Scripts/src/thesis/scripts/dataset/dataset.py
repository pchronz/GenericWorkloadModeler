'''
Created on Jul 8, 2011

@author: Claudio
'''
import time
import csv
from calendar import weekday
from numpy import array, float64
from thesis.scripts.dataset.epoch import toepoch
def weeklydataset(filesource, label_index):
    
    reader = csv.reader(open(filesource, "rb"), delimiter= ';', quotechar = '"')
    indexmon = list()
    indextue = list()
    indexwed = list()
    indexthu = list()
    indexfri = list()
    indexsat = list()
    indexsun = list()
    weekindex = [indexmon, indextue, indexwed, indexthu, indexfri, indexsat, indexsun]
    label = list()
    # vectorDataSet will contain the request from monday to friday 
    #and the request are sorted by the hour
    vectorDataSet = list()
    
    for row in reader:
        standarddate = time.gmtime(float(row[0]))
        weekindex[weekday(standarddate[0], standarddate[1], standarddate[2])].append(row)
    
    for index in weekindex:
        sortedindex = sorted(index, key=lambda hour : time.gmtime(float(hour[0]))[4])
        sortedindex = sorted(sortedindex, key=lambda hour : time.gmtime(float(hour[0]))[3])
        for index2 in sortedindex:
            test = list()
            train = list()
            for item in range(len(index2)):
                if item in label_index:
                    test.append(index2[item])
                else:
                    train.append(index2[item])
            label.append(test)
            vectorDataSet.append(train)
    
    
#    vectorDataSet = array(vectorDataSet, dtype=float64)
    return vectorDataSet, label

def weeklydataset_sg_ndata(filesource, label_index):
    
    trainreader = csv.reader(open(filesource, "rb"), delimiter= ',', quotechar = '"')
    indexmon = list()
    indextue = list()
    indexwed = list()
    indexthu = list()
    indexfri = list()
    indexsat = list()
    indexsun = list()
    weekindex = [indexmon, indextue, indexwed, indexthu, indexfri, indexsat, indexsun]
    
    # vectorDataSet will contain the request from monday to sunday 
    #and the request are sorted by the hour
    count = 0
    nline = 0
    for row in trainreader:
        nline += 1
        try:
            count = len(row)
            rawdate = row[1]
            standarddate = time.gmtime(toepoch(rawdate))
            row[5] = float(row[5])
            row[6] = float(row[6])
            row[7] = float(row[7])
            weekindex[weekday(standarddate[0], standarddate[1], standarddate[2])].append(row)
        except:
            print nline
            
    
    vectorDataSet = []
    for column in range(count - len(label_index)):
        vectorDataSet.append(list())
    
    label = list()
    for column in range(len(label_index)):
        label.append(list())
    for index in weekindex:
        sortedindex = sorted(index, key=lambda hour : time.gmtime(toepoch(hour[1]))[4])
        sortedindex = sorted(sortedindex, key=lambda hour : time.gmtime(toepoch(hour[1]))[3])
#        vectorDataSet.append(sortedindex)
        for index2 in sortedindex:
            labelcounter = 0
            vectorcounter = 0
            for item in range(len(index2)):
                if item in label_index:
                    label[labelcounter].append(index2[item])
                    labelcounter = labelcounter+1
                else:
                    vectorDataSet[vectorcounter].append(index2[item])
                    vectorcounter = vectorcounter+1
#                    train.append(index2[item])
    
     
    return vectorDataSet, label

def weeklydataset_shogun(filesource, label_index):
    
    trainreader = csv.reader(open(filesource, "rb"), delimiter= ',', quotechar = '"')
    indexmon = list()
    indextue = list()
    indexwed = list()
    indexthu = list()
    indexfri = list()
    indexsat = list()
    indexsun = list()
    weekindex = [indexmon, indextue, indexwed, indexthu, indexfri, indexsat, indexsun]
    
    # vectorDataSet will contain the request from monday to friday 
    #and the request are sorted by the hour
    count = 0
    for row in trainreader:
        count = len(row)
        standarddate = time.gmtime(float(row[0]))
#        print standarddate
        weekindex[weekday(standarddate[0], standarddate[1], standarddate[2])].append(row)
    
    vectorDataSet = list()
    for column in range(count - len(label_index)):
        vectorDataSet.append(list())
    
    label = list()
    for column in range(len(label_index)):
        label.append(list())
    for index in weekindex:
        sortedindex = sorted(index, key=lambda hour : time.gmtime(float(hour[0]))[4])
        sortedindex = sorted(sortedindex, key=lambda hour : time.gmtime(float(hour[0]))[3])
        for index2 in sortedindex:
            labelcounter = 0
            vectorcounter = 0
            for item in range(len(index2)):
                if item in label_index:
                    label[labelcounter].append(index2[item])
                    labelcounter = labelcounter+1
                else:
                    vectorDataSet[vectorcounter].append(index2[item])
                    vectorcounter = vectorcounter+1
#                    train.append(index2[item])
    
    vectorDataSet = array(vectorDataSet, dtype=float64)
    return vectorDataSet, label

def dataset(filesource, label_index):
    
    reader = csv.reader(open(filesource, "rb"), delimiter= ',', quotechar = '"')
    train = list()
    label = list()
    for row in reader:
        tmptrain = list()
        tmptest = list()
        for item in range(len(row)):
            if item in label_index:
                tmptest.append(row[item])
            else:
                tmptrain.append(row[item])
        train.append(tmptrain)
        label.append(tmptest)
    
    return train, label