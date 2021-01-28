import numpy as np
#from VAR import *

def Logreturns(x):
    return np.log(x) - np.log(x.shift(1))

def AssetRisk(x):
    '''
    Portfolio returns variance as risk
    '''
    return np.var(x)

def CovarianceMatrix(x):
    '''
    Compute Covariance matrix for log returns normalize *252
    '''
    return x.cov()* 252

def ExpectedReturn(x):
    return x.mean()