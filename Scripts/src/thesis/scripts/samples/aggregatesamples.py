'''
Created on Jul 18, 2011

@author: work
'''
import time
from matplotlib.pyplot import figure, show
from thesis.scripts.dataset.epoch import toepoch
import datetime
from numpy import zeros
def aggregatebymins(points):
    
    mon = [0]*1440
    tue = [0]*1440
    wed = [0]*1440
    thu = [0]*1440
    fri = [0]*1440
    sat = [0]*1440
    sun = [0]*1440
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in points:
#        standarddate = time.gmtime(float(line[0]))
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        req = week[dweek][(hour *60) + min]
        req += 1
        week[dweek][(hour *60) + min] = req
        
    target = mon + tue + wed + thu + fri + sat + sun
    input = [x for x in range(1440*7)]
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(input, target, linewidths = 1.0)
    ax1.axis([0,10080,0,1900])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return input, target

def aggregatebymins_sg(timestamps):
    
    mon = [0]*1440
    tue = [0]*1440
    wed = [0]*1440
    thu = [0]*1440
    fri = [0]*1440
    sat = [0]*1440
    sun = [0]*1440
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        req = week[dweek][(hour *60) + min]
        req += 1
        week[dweek][(hour *60) + min] = req
        
    for wday in week:
        for m in wday:
            m = m/53
    
    target = mon + tue + wed + thu + fri
    input = [x for x in range(1440*5)]
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(input, target)
    ax1.axis([0,7200,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return input, target

def aggregateby30sec_sg(timestamps):
    
    mon = [0]*2880
    tue = [0]*2880
    wed = [0]*2880
    thu = [0]*2880
    fri = [0]*2880
    sat = [0]*2880
    sun = [0]*2880
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]

        week[dweek][(hour *120) + (2*min)] += 1
        
    for wday in week:
        for m in wday:
            m = m/53
    
    target = mon + tue + wed + thu + fri + sat + sun 
    input = [x for x in range(2880*7)]
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(input, target)
#    ax1.axis([0,7200,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return input, target

def aggregateby10mins_sg(timestamps):
    
    mon = [0]*144
    tue = [0]*144
    wed = [0]*144
    thu = [0]*144
    fri = [0]*144
    sat = [0]*144
    sun = [0]*144
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        week[dweek][(hour *6) + min/10] +=1
        
    target = mon + tue + wed + thu + fri + sat + sun
    for i in range(len(target)):
        if target[i] > 500:
            target[i] = 500
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(144*7)]
    y = week
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y)
    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return x, target

def aggregateby10mins_sg_mean(timestamps, numcluster):
     
    mon = [0]*144
    tue = [0]*144
    wed = [0]*144
    thu = [0]*144
    fri = [0]*144
    sat = [0]*144
    sun = [0]*144
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        week[dweek][(hour *6) + min/10] +=1
        
    for wday in week:
        for m in wday:
            m = m/53
    
    target = mon + tue + wed + thu + fri + sat + sun
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(144*7)]
    y = week
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y)
    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    fig.savefig("aggregation_cluster_%d" % (numcluster), format='png')
    
    return x, target

def aggregateby10mins_sg_ndata(timestamps, numbercluster):
     
    mon = [0]*144
    tue = [0]*144
    wed = [0]*144
    thu = [0]*144
    fri = [0]*144
    sat = [0]*144
    sun = [0]*144
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(toepoch(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        week[dweek][(hour *6) + min/10] +=1
        
#    for wday in week:
#        for m in wday:
#            m = m/53
    
    target = mon + tue + wed + thu + fri + sat + sun
#    for i in range(len(target)):
#        if target[i] > 500:
#            target[i] = 500
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(144*7)]
    y = week
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y)
    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    fig.savefig("aggregation_cluster_%d" % (numbercluster), format='png')
#    show()
    
    return x, target
def aggregatebymins_sg_ndata(timestamps):
     
    mon = [0]*1440
    tue = [0]*1440
    wed = [0]*1440
    thu = [0]*1440
    fri = [0]*1440
    sat = [0]*1440
    sun = [0]*1440
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(toepoch(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        week[dweek][(hour *60) + min] +=1
        
#    for wday in week:
#        for m in wday:
#            m = m/53
    
    target = mon + tue + wed + thu + fri + sat + sun
#    for i in range(len(target)):
#        if target[i] > 500:
#            target[i] = 500
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(1440*7)]
    y = week
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y)
    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return x, target

def aggregateby30sec_sg_ndata(timestamps):
     
    mon = [0]*2880
    tue = [0]*2880
    wed = [0]*2880
    thu = [0]*2880
    fri = [0]*2880
    sat = [0]*2880
    sun = [0]*2880
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(toepoch(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        
        week[dweek][(hour *120) + (2*min)] +=1
        
#    for wday in week:
#        for m in wday:
#            m = m/53
    
    target = mon + tue + wed + thu + fri + sat + sun
#    for i in range(len(target)):
#        if target[i] > 500:
#            target[i] = 500
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(2880*7)]
    y = week
    
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y)
    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    show()
    
    return x, target

def aggregateby10mins_sg_mcmc(timestamps, numcluster):
    mon = []
    [mon.append([0]*53) for i in range(144)]
    tue = []
    [tue.append([0]*53) for i in range(144)]
    wed = []
    [wed.append([0]*53) for i in range(144)]
    thu = []
    [thu.append([0]*53) for i in range(144)]
    fri = []
    [fri.append([0]*53) for i in range(144)]
    sat = []
    [sat.append([0]*53) for i in range(144)]
    sun = []
    [sun.append([0]*53) for i in range(144)]
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    for line in timestamps:
        standarddate = time.gmtime(float(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
        weeknumber = datetime.datetime(standarddate[0], standarddate[1], standarddate[2],0,0).isocalendar()[1]
        week[dweek][(hour *6) + min/10] [weeknumber] +=1
    
    target = mon + tue + wed + thu + fri + sat + sun
    x = [inp for inp in range(144*7)]
    tmp = zip(*target)
    y1 = tmp[0]
    y2 = tmp[1]
    
    print "len x = %f" % len(x)
    print "len y1 = %f" % len(y1)
    print "len y2 = %f" % len(y2)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y1, c='b')
    ax1.scatter(x, y2, c='r')
#    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    fig.savefig("aggregation_cluster_%d" % (numcluster), format='png')
    
    return target

def aggregateby10mins_sg_mcmc_ndata(timestamps, numbercluster):
    mon = []
    [mon.append([0]*2) for i in range(144)]
    tue = []
    [tue.append([0]*2) for i in range(144)]
    wed = []
    [wed.append([0]*2) for i in range(144)]
    thu = []
    [thu.append([0]*2) for i in range(144)]
    fri = []
    [fri.append([0]*2) for i in range(144)]
    sat = []
    [sat.append([0]*2) for i in range(144)]
    sun = []
    [sun.append([0]*2) for i in range(144)]
    week = [mon, tue, wed, thu, fri, sat, sun]
    
    
    for line in timestamps:
        standarddate = time.gmtime(toepoch(line))
        dweek = standarddate[6]
        hour = standarddate[3]
        min = standarddate[4]
#        weeknumber = datetime.datetime(standarddate[0], standarddate[1], standarddate[2],0,0).isocalendar()[1]
        if (standarddate[2] <= 7):
            week[dweek][(hour *6) + min/10] [0] +=1
        else:
            week[dweek][(hour *6) + min/10] [1] +=1
    
    target = mon + tue + wed + thu + fri + sat + sun
    
    input = [inp for inp in range(144*5)]
    
    x = [inp for inp in range(144*7)]
    tmp = zip(*target)
    y1 = tmp[0]
    y2 = tmp[1]
    
    print "len x = %f" % len(x)
    print "len y1 = %f" % len(y1)
    print "len y2 = %f" % len(y2)
    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x, y1, c='b')
    ax1.scatter(x, y2, c='r')
#    ax1.axis([0,max(x)+10,0,max(target)+100])
    ax1.set_xlabel('minute of the week')
    ax1.set_ylabel('Number of requests')
    fig.savefig("aggregation_cluster_%d" % (numbercluster), format='png')
    
    return target