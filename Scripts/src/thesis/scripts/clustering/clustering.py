'''
Created on Jul 17, 2011

@author: work
'''
from Pycluster import kcluster
from Pycluster import clustercentroids
from matplotlib.pyplot import figure, show
from numpy import array, float64

def create_clustered_samples_ndata(points, nclusters, transpose):
    
#    print points[5:7]
    used_points = array(points[6:8], dtype = float64)
    labels, error, nfound= kcluster(used_points, nclusters, None, None, transpose, npass=1, method='a', dist='e', initialid=None)
    
    cdata, cmask = clustercentroids(used_points, None, labels, 'a', transpose)
    
    print cdata
    
    
    
    clusteredpoints = list()
    
    for i in range(nclusters):
        clusteredpoints.append(list())
    if transpose == 0:    
        for index in range(len(points)):
            clusteredpoints[labels[index]].append(points[index])
        return clusteredpoints, cdata
    else:
        for i in range(len(clusteredpoints)):
            for types in range(len(points)):
                clusteredpoints[i].append(list())
        for index in range(len(labels)):
            for item in range(len(points)):
                clusteredpoints[labels[index]][item].append(points[item][index])
        #print clusters and some element
#        x = cdata[1]
#        y = cdata[2]
        
#        fig = figure()
#        ax1 = fig.add_subplot(1,1,1)
#        ax1.scatter(x, y, c='r')
#        ax1.axis([0,max(x)+1,0,max(y)+1])
#        ax1.set_xlabel('number of bodies')
#        ax1.set_ylabel('number of steps')
        
#        x0 = clusteredpoints[0][2]
#        y0 = clusteredpoints[0][3]
#        
#        x1 = clusteredpoints[1][1]
#        y1 = clusteredpoints[1][2]
#        
#        x2 = clusteredpoints[2][1]
#        y2 = clusteredpoints[2][2]
#        
#        x3 = clusteredpoints[3][1]
#        y3 = clusteredpoints[3][2]
#        
#        x4 = clusteredpoints[4][1]
#        y4 = clusteredpoints[4][2]
#        
#        x5 = clusteredpoints[5][1]
#        y5 = clusteredpoints[5][2]
#        
#        ax1.scatter(x0[1:20],y0[1:20], marker='s')
#        ax1.scatter(x1[1:20],y1[1:20], marker='^')
#        ax1.scatter(x2[1:15],y2[1:15], marker='<')
#        ax1.scatter(x3[1:15],y3[1:15], marker='>')
#        ax1.scatter(x4[1:15],y4[1:15], marker='p')
#        ax1.scatter(x5[1:15],y5[1:15], marker='8')
#        show()
        return clusteredpoints, cdata

def create_clustered_samples(points, nclusters, transpose):
    
    print points[1:6]
    labels, error, nfound= kcluster(points[1:4], nclusters, None, None, transpose, npass=1, method='a', dist='e', initialid=None)
    
    cdata, cmask = clustercentroids(points[1:4], None, labels, 'a', transpose)
    
    print cdata
    
    
    
    clusteredpoints = list()
    
    for i in range(nclusters):
        clusteredpoints.append(list())
    if transpose == 0:    
        for index in range(len(points)):
            clusteredpoints[labels[index]].append(points[index])
        return clusteredpoints, cdata
    else:
        for i in range(len(clusteredpoints)):
            for types in range(len(points)):
                clusteredpoints[i].append(list())
        for index in range(len(labels)):
            for item in range(len(points)):
                clusteredpoints[labels[index]][item].append(points[item][index])
        #print clusters and some element
        x = cdata[1]
        y = cdata[2]
        
#        fig = figure()
#        ax1 = fig.add_subplot(1,1,1)
#        ax1.scatter(x, y, c='r')
#        ax1.axis([0,max(x)+1,0,max(y)+1])
#        ax1.set_xlabel('number of bodies')
#        ax1.set_ylabel('number of steps')
        
#        x0 = clusteredpoints[0][2]
#        y0 = clusteredpoints[0][3]
#        
#        x1 = clusteredpoints[1][1]
#        y1 = clusteredpoints[1][2]
#        
#        x2 = clusteredpoints[2][1]
#        y2 = clusteredpoints[2][2]
#        
#        x3 = clusteredpoints[3][1]
#        y3 = clusteredpoints[3][2]
#        
#        x4 = clusteredpoints[4][1]
#        y4 = clusteredpoints[4][2]
#        
#        x5 = clusteredpoints[5][1]
#        y5 = clusteredpoints[5][2]
#        
#        ax1.scatter(x0[1:20],y0[1:20], marker='s')
#        ax1.scatter(x1[1:20],y1[1:20], marker='^')
#        ax1.scatter(x2[1:15],y2[1:15], marker='<')
#        ax1.scatter(x3[1:15],y3[1:15], marker='>')
#        ax1.scatter(x4[1:15],y4[1:15], marker='p')
#        ax1.scatter(x5[1:15],y5[1:15], marker='8')
#        show()
        return clusteredpoints, cdata
    
def classify_input(points, cdata, transpose):
    
    clusteredpoints = list()
    
    for i in range(len(cdata[0])):
        clusteredpoints.append(list())
    if transpose == 1:
        from shogun.Distance import EuclidianDistance
        from shogun.Features import RealFeatures
        from numpy  import min, where
        
        obs = RealFeatures(points)
        centroids = RealFeatures(cdata)
        
        distance = EuclidianDistance(obs, centroids)
        
#        out1 = distance.get_distance_matrix()
        distance.init(obs, centroids)
        out = distance.get_distance_matrix()
        
        labels = list()

        for i in range(len(out)):
#            dist = distances[:len(points),i]
            dist = zip(out)[i]
            minimum = min(dist)
            itemindex=where(dist==minimum)
            labels.append(itemindex[1][0])
        
        print len(points)
        print len(points[0])
        for i in range(len(clusteredpoints)):
            for types in range(len(points)):
                clusteredpoints[i].append(list())
        for index in range(len(labels)):
            for item in range(len(points)):
                clusteredpoints[labels[index]][item].append(points[item][index])
        return clusteredpoints, labels