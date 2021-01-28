from Returns import *
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

solvers.options['show_progress'] = False

def convert_portfolios(portfolios):
    '''
    Takes in a cvxopt matrix and returns list
    '''
    portfolio_list = []
    for portfolio in portfolios:
        temp = np.array(portfolio)
        portfolio_list.append(temp[0].tolist())

    return portfolio_list

def OptimalPortfolio(Returns):
    '''
    Return Optimal Portfolio given a matrix of return
    '''
    n = len(Returns)
    #number of assets on dataframe
    returns = np.asmatrix(Returns)

    #Why 100?
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    #Covert to covariance matrix opt matrix
    S = opt.matrix(np.cov(Returns))

    pbar = opt.matrix(np.mean(Returns,axis=1))
    print("Portfolio Mean:", pbar)

    #Create Constraint matrices
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate Effecient Frontier weights using Quadratic Programming

    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    portfolio_list = convert_portfolios(portfolios)
    print(portfolio_list)

    #Compute Risk and Return for Frontier

    returns = [blas.dot(pbar,x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

    # The second degree of polynmials for the frontier
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2]/m1[0])
    #Calculate Optimal Portfolio
    wt = solvers.qp(opt.matrix(x1 * S),-pbar,G,h,A,b)['x']
    print("weight of, optimal portfolio:", wt)
    return np.asarray(wt), returns, risks, portfolio_list

def CovarianceMeanPortfolios(covariances, mean_returns):
    n = len(mean_returns)
    N = 100
    mus = [10**(5.0* t/N -1.0) for t in range(N)]

    S = opt.matrix(covariances)  # how to convert array to matrix?  
    pbar = opt.matrix(mean_returns)  # how to convert array to matrix?

    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    #portfolios = [solvers.qp(mu*S, -pbar, G, h)['x'] for mu in mus]
    port_list = convert_portfolios(portfolios)
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  

    return np.asarray(wt), frontier_returns, risks, port_list