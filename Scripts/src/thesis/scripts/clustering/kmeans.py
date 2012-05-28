'''
Created on 07/mag/2012

@author: claudio
'''
from collections import Counter, defaultdict
from numpy import array, random
from numpy.ma.core import absolute, sqrt
from thesis.scripts.dataset.dataset import weeklydataset
from operator import itemgetter
import csv
import Levenshtein
from time import time

def distance(centroid, observation):
    dists = []
    for i in range(0, len(centroid)):
#        if i == 1:
#            dists.append(pow(absolute(int(centroid[i]) - int(observation[i])),2))
#        else:
        if i != 1:
            dists.append(pow(Levenshtein.distance(centroid[i], observation[i]),2))
    
    return sqrt(sum(dists))

def kmeans(data, num_clusters):
    
    centroids = []
    clusters = []
    centroids_val = []
    par1 = []
    par2 = []
    par3 = []
    par4 = []
#    exec_output = open("exec_output_dict_mod.txt", "w")
    # initialization of the k centroids and clusters with void lists
    for i in range(0,num_clusters):
        par1.append(defaultdict(int))
        par2.append(defaultdict(int))
        par3.append(defaultdict(int))
        par4.append(defaultdict(int))
        clusters.append([])
        cent_tmp = []
        j = 1
        
        while j < len(data[1]):
            init_mean = data[random.randint(0, len(data)-1)]
            val = init_mean[j]
            if (not centroids_val) or not(val in centroids_val):
                cent_tmp.append(val)
                centroids_val.append(val)
                j = j + 1
        centroids.append(cent_tmp)
        
    
    print centroids
    # iterative step
    for observation in data:
        #take the value from the observations)
        cluster_errors = []
        
        #sum the levenshtein distances among the values in each cluster with the observed value
#        exec_output.write("Centroid distance computation: \n")
#        start_time = time() 
        for c in centroids:
            cluster_errors.append(distance(c, observation[1:len(observation)]))
#        end_time = time()
#        exec_output.write("%10.3f \n" % (end_time - start_time))
        
        
        #discover the cluster with the minimum error
#        exec_output.write("Minimum distance discovery: \n")
#        start_time = time() 
        min_error = min(cluster_errors)
        min_index = cluster_errors.index(min_error, )
#        end_time = time()
#        exec_output.write("%10.3f \n" % (end_time - start_time))
        
        #assign the observed value to the cluster, update the centroids value
        clusters[min_index].append(observation)
        
        
        #update the centroid
#        exec_output.write("Update centroid: \n")
#        start_time = time()
#        types = array(clusters[min_index]).transpose()
        
#        par1, num_par1 = Counter(types[1]).most_common(1)[0]
#        par2, num_par2 = Counter(types[2]).most_common(1)[0]
#        par3, num_par3 = Counter(types[3]).most_common(1)[0]
#        par4, num_par4 = Counter(types[4]).most_common(1)[0]

#        par1 = defaultdict(int)
#        par2 = defaultdict(int)
#        par3 = defaultdict(int)
#        par4 = defaultdict(int)
        
#        for item in clusters[min_index]:
        par1[min_index][observation[1]] += 1
        par2[min_index][observation[2]] += 1
        par3[min_index][observation[3]] += 1
        par4[min_index][observation[4]] += 1
            
        
        centroids[min_index] = [max(par1[min_index].iteritems(), key=itemgetter(1))[0], max(par2[min_index].iteritems(), key=itemgetter(1))[0], max(par3[min_index].iteritems(), key=itemgetter(1))[0], max(par4[min_index].iteritems(), key=itemgetter(1))[0]]
#        end_time = time()
#        exec_output.write("%10.3f \n" % (end_time - start_time))
        
#    exec_output.close()
    return centroids, clusters       
        

def test():
    
#    data = [["claudio","Di Cosmo"], ["claudino", "Cosimino"], ["fabio", "Melillo"], ["fabietto", "Mellillo"], ["angelo", "Furno"], ["angioletto", "Furnetto"], ["antonio", "Cuomo"], ["antoniuccio", "Cuomuccio"], ["marcangelo", "Frunillo"]]
    X, label = weeklydataset('/home/claudio/Workloads/WmProxyWL/nlog.csv', [])
    start_time = time()
    centroids, clusters = kmeans(X, 4)
    end_time = time()
    
    print end_time - start_time
    
    print centroids
    print len(clusters[0])
    print len(clusters[1])
    print len(clusters[2])
    print len(clusters[3])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster0.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 0", len(clusters[0])])
    results.writerows(clusters[0])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster1.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 1", len(clusters[1])])
    results.writerows(clusters[1])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster2.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 2", len(clusters[2])])
    results.writerows(clusters[2])
    
    results = csv.writer(open("/home/claudio/Workloads/WmProxyWL/cluster3.csv", "wb"), delimiter=";")
    
    results.writerow(["Cluster 3", len(clusters[3])])
    results.writerows(clusters[3])
    
     
if __name__ == '__main__':
    test()        