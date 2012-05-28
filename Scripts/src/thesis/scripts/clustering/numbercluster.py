'''
Created on Jul 18, 2011

@author: work
'''
#from Pycluster import clustercentroids, kcluster
from kmeans import kmeans
from numpy import matrix, float64 
from thesis.scripts.dataset.dataset import weeklydataset
import matplotlib.pyplot as plt


#[X, label] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_mod.csv', [0])
X, label = weeklydataset('/home/claudio/Workloads/WmProxyWL/log_mod.csv', [])
#X = open('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu.csv',)

K = range(2,10)

labels = list()
error = list()
nfound = list()
cdata = list()
cmask = list()
#param = X[5:8]
parameters = matrix(X)
for k in K:
#    tmplabels, tmperror, tmpnfound = kcluster(parameters, nclusters=k, mask=None, weight=None, transpose=1, npass=1, method='a', dist='e', initialid=None)
#    tmpcdata, tmpcmask = clustercentroids(parameters, None, tmplabels, 'a', 1)
    tmperror, tmp_cluster = kmeans(X, k)
#    labels.append(tmplabels)
    error.append(tmperror)
#    nfound.append(tmpnfound)
#    cdata.append(tmpcdata)
#    cmask.append(tmpcmask)
    
avgWithinSS = [err/parameters.shape[0] for err in error]
kIdx=1
##### plot ###

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()
