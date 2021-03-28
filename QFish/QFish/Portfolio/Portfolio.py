"""
Class with Portfolio Applications
"""
import numpy as np
from cvxpy import *
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
from tabulate import tabulate



solvers.options['show_progress'] = False

"""convert portfolios to list"""
def  convert(portfolios):
    portfolio_list = []
    for portfolio in portfolios:
        temp = np.array(portfolio)
        portfolio_list.append(temp[0].tolist())

    return portfolio_list

class Portfolio:
    """Initialize the assets as a tuple"""
    def __init__(self, mean, covariance, riskfree = 0.005, stdDev,frequency=252):
        #self.returns = returns
        self.mean = mean
        self.covariance = covariance
        self.risk_free_rate = riskfree
        self.stdDev = stdDev
        self.frequency = frequency
        self.names = list(mean.index)
        self.num_stocks = len(self.mean)

        #Place holders for optimized values
        self.weights = None

        #compute return for assets
        #covariance matrix
        #weights assignment
    def meanVarPortfolio(self):
        n = self.num_stocks

        N = self.num_stocks
        mus = [10**(5.0* t/N -1.0) for t in range(N)]
        #covariance matrix
        cov = opt.matrix(self.covariance.to_numpy())
        #print(self.covariance)

        #means
        pbar = opt.matrix(self.mean.to_numpy())
        
        #optimization constraints

        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        #efficient frontier weights using QP
        portfolios = [solvers.qp(mu*cov, -pbar, G, h, A, b)['x'] for mu in mus]
        portfolio_list = convert(portfolios)
        #print(portfolio_list)

        #risk and return for the frontier
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, cov*x)) for x in portfolios] 

        """next compute sharpe ratio"""

        # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(frontier_returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * cov), -pbar, G, h, A, b)['x']
        return np.asarray(wt), frontier_returns, risks, portfolio_list
        #compute value at Risk of The portfolio
    def sharpeRatio(self):
        return (self.mean - self.risk_free_rate)/self.stdDev
    
    def plotEffecientFrontier(self):
        weights, returns, risk, portfolios = self.meanVarPortfolio()
        print(tabulate([self.names],[portfolios]))
        plt.ylabel('mean')
        plt.xlabel('std')
        plt.plot(risk, returns,'y-o',linestyle="-",color="blue",lw=2,label="Efficient Frontier")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.show()
    
    def plot_optimal_portfolios(self):
        """
        plot markers for optimized portfolios

        min volatility-> min variance
        maximum sharpe ratio
        """
        return False
