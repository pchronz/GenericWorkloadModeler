'''
Created on Jul 12, 2011

@author: Claudio
'''
from shogun.Distance import EuclidianDistance
from shogun.Features import RealFeatures
from shogun.Clustering import Hierarchical
from shogun.Clustering import KMeans
from shogun.Library import Math_init_random

def hierarchical(train,merges=3):

    feats_train=RealFeatures(train)
    distance=EuclidianDistance(feats_train, feats_train)

    hierarchical=Hierarchical(merges, distance)
    hierarchical.train()

    out_distance = hierarchical.get_merge_distances()
    out_cluster = hierarchical.get_cluster_pairs()

    return hierarchical,out_distance,out_cluster 


def kmeans (train,k=3):

    Math_init_random(17)

    feats_train=RealFeatures(train)
    distance=EuclidianDistance(feats_train, feats_train)

    kmeans=KMeans(k, distance)
    kmeans.train()

    out_centers = kmeans.get_cluster_centers()
    kmeans.get_radiuses()

    return out_centers, kmeans