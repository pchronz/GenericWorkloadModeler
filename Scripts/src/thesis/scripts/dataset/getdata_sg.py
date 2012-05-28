'''
Created on Jul 12, 2011

@author: Claudio
'''

from thesis.scripts.dataset.dataset import weeklydataset_shogun
#from thesis.scripts.clustering.clustering import hierarchical
from thesis.scripts.svr.svr import svr
from numpy import double, fromfile

def getalldata():
    
    train = weeklydataset_shogun("/home/work/Projects/EclipseProjects/thesis/Scripts/train_sg.csv")
    test = weeklydataset_shogun("/home/work/Projects/EclipseProjects/thesis/Scripts/test_sg.csv")
    
    train_label = fromfile("/home/work/Projects/EclipseProjects/thesis/Scripts/train_label.csv", dtype=double, sep=' ')
    test_label = fromfile("/home/work/Projects/EclipseProjects/thesis/Scripts/test_label.csv", dtype=double, sep=' ')
    
    '''
    train = load_numbers("/home/work/Projects/EclipseProjects/thesis/Scripts/fm_train_real.dat")
    test = load_numbers("/home/work/Projects/EclipseProjects/thesis/Scripts/fm_test_real.dat")
    train_label = load_labels("/home/work/Projects/EclipseProjects/thesis/Scripts/label_train_multiclass.dat")
    '''
    return train, test, train_label, test_label

def load_numbers(filename):
    matrix=fromfile(filename, sep=' ')
    # whole matrix is 1-dim now, so reshape
    matrix=matrix.reshape(2, len(matrix)/2)
    #matrix=matrix.reshape(len(matrix)/2, 2)

    return matrix


def load_dna(filename):
    fh=open(filename, 'r');
    matrix=[]

    for line in fh:
        matrix.append(line[:-1])

    return matrix


def load_cubes(filename):
    fh=open(filename, 'r');
    matrix=[]

    for line in fh:
        matrix.append(line.split(' ')[0][:-1])

    fh.close()

    return matrix


def load_labels(filename):
    return fromfile(filename, dtype=double, sep=' ')