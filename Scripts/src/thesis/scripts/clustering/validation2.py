'''
Created on Jul 15, 2011

@author: work
'''
import numpy as np
from thesis.scripts.dataset.dataset import weeklydataset_shogun
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
from matplotlib import cm

# load the iris dataset
#fName = 'C:\\Python26\\Lib\\site-packages\\scipy\\spatial\\tests\\iris.txt'
#fp = open(fName)
#X = np.loadtxt(fp)
#fp.close()
[X, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu.csv', [0])
##### cluster data into K=1..10 clusters #####
K = range(1,10)

# scipy.cluster.vq.kmeans
KM = [kmeans(X,k) for k in K]
centroids = [cent for (cent,var) in KM]   # cluster centroids
#avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares

# alternative: scipy.cluster.vq.vq
#Z = [vq(X,cent) for cent in centroids]
#avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]

# alternative: scipy.spatial.distance.cdist
D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]
#tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
#totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
#betweenss = totss - tot_withinss          # The between-cluster sum of squares

##### plot ###
kIdx = 5

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

##### plots #####
#clr = cm.spectral( np.linspace(0,1,10) ).tolist()
#mrk = 'os^p<dvh8>+x.'

# elbow curve
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(KK, betweenss/totss*100, 'b*-')
#ax.set_ylim((0,100))
#plt.grid(True)
#plt.xlabel('Number of clusters')
#plt.ylabel('Percentage of variance explained (%)')
#plt.title('Elbow for KMeans clustering')

plt.show()
