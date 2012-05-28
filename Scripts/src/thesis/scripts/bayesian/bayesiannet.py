'''
Created on Jul 24, 2011

@author: work
'''
from pebl import data, result
from pebl.learner import greedy, simanneal
from numpy import matrix, array, empty
from pebl.data import Dataset
import thesis.scripts.bayesian.Requestmodel as requestmodel
from pymc.Matplot import plot
from pymc import MCMC
from thesis.scripts.examples import main_ex
def bayesian_structure():
    
#    dataset = data.fromfile("/home/work/pebl-tutorial-data2.txt")
#    dataset = Dataset(points.transpose(), None, None, ['in','tar'], None, None)
    dataset = data.fromfile('/home/work/Projects/EclipseProjects/thesis/Scripts/elements.csv')
    dataset.discretize()
#    dataset.discretize()
#    learners = [ greedy.GreedyLearner(dataset, max_iterations=1000000) for i in range(5) ] + \
#    [ simanneal.SimulatedAnnealingLearner(dataset) for i in range(5) ]
#    merged_result = result.merge([learner.run() for learner in learners])
    learner = greedy.GreedyLearner(dataset)
    merged_result = learner.run()
    merged_result.tohtml("example-result")
    
def bayesian_model():
    
    
    
#    requestmodel.setrequest(traintarget)
    
    model = MCMC(requestmodel)
    
#    plot(model) 
#    model.isample(iter=10000, burn=1000, thin=10)
#    print model.trace('l')[:]
    return model