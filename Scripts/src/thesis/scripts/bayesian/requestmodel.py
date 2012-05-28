'''
Created on Jul 29, 2011

@author: work
'''
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, MCMC
from pymc.Matplot import plot
from thesis.scripts.examples.main_ex import main_mcmc
from numpy import matrix, array, empty, int32

input, target = main_mcmc()

request_lbl = matrix(target, dtype=int32)

s = DiscreteUniform('s', lower=0, upper=1008, doc='Switchpoint[10minutes]')

e = Exponential('e', beta=1)

l = Exponential('l', beta=1)

@deterministic(plot=False)
def r(s=s, e=e, l=l):
    """ Concatenate Poisson means """
    out = empty(len(request_lbl))
    out[:s] = e
    out[s:] = l
    return out

R = Poisson('R', mu=r, value=request_lbl, observed=True)