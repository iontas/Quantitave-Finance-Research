import Returns as rt
import VaR
import yfinance as yf
import pandas as pd
from typing import List
import Portfolio as pt
import matplotlib.pyplot as plt

stocks = ['AAPL', 'AMZN', 'GOOG', 'JNJ', 'JPM']

#rt.getStocks(stocks,'1mo')
tickers = yf.Tickers(stocks)
"""returns a named Tuple if Ticker Objects"""
history = tickers.history(period="3mo")['Close']
"""dataframe"""
df = pd.DataFrame(data=history)
rt = rt.Returns(df)

#print(rt.descriptiveStats())
#print(df)
#var = VaR.VaR(rt.meanReturn(),rt.stdvReturns(),0.95)
mean = rt.meanReturn()
covariance = rt.covReturns()
portfolio = pt.Portfolio(mean, covariance)

portfolio.plotEffecientFrontier()

#weights, returns, risk, portfolios = portfolio.meanVarPortfolio()
#portfolio.meanVarPortfolio()
#print(portfolios)

#print(rt.logReturns())
#print(portfolio.meanVarPortfolio())
#print(var.valueAtRisk())
#print(var.conditionalVaR())