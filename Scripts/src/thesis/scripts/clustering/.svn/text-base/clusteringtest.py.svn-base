'''
Created on Jul 19, 2011

@author: work
'''
import sys
from thesis.scripts.dataset.dataset import weeklydataset_shogun
from thesis.scripts.dataset.dataset import weeklydataset
from thesis.scripts.clustering.clustering import create_clustered_samples
from thesis.scripts.clustering.clustering import classify_input
from thesis.scripts.samples.traintest import traintest

def main(argv):
    #[train, labels] = weeklydataset('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_500.csv', [])
    [points, labels] = weeklydataset_shogun('/home/work/Projects/EclipseProjects/thesis/Scripts/cpu_500.csv', [])
    
    clusteredpoints, cdata = create_clustered_samples(points, 5, 1)
#    print len(clusteredpoints[0][2])
    
    train ,test = traintest(clusteredpoints[0], 20, 1)
    
    
#    [testclusterpoint, testlbl] = classify_input(test, cdata, 1)
    
    
    
    
#    print clusteredpoints[0]

if __name__ == '__main__':
    main(sys.argv[1:])