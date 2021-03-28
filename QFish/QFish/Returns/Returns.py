"""
returns class
"""
import yfinance as yf
import pandas as pd
from typing import List, Tuple
import numpy as np
from tabulate import tabulate

class Returns:
    """tuple of assets"""
    def __init__(self, data):
        #self.data= pd.DataFrame(data)
        self.data = data
        """data"""
    
    def logReturns(self):
        return np.log(self.data) - np.log(self.data.shift(1))

    def normReturns(self):
        mu = self.meanReturn()
        std = self.stdvReturns()
        result = (self.logReturns()-mu)/std
        return result

    def varianceReturns(self):
        return self.logReturns().var()
    
    def stdvReturns(self):
        return self.logReturns().std()
    
    def covReturns(self):
        return self.logReturns().cov()
    
    def corrReturn(self,method):
        self.method = method
        return self.logReturns().corr(method=self.method)
    
    def meanReturn(self):
        return self.logReturns().mean()
    
    def skewReturn(self):
        return self.logReturns().skew()
    
    def kurtosisReturn(self):
        return self.logReturns().kurt()
    
    def descriptiveStats(self):
        table = [["returns variance",self.varianceReturns()],
        ["standard deviation", self.stdvReturns()],
        ["mean returns",self.meanReturn()],
        ["returns skew",self.skewReturn()],
        ["return kurtosis", self.kurtosisReturn()]]
        return tabulate(table)


    
    """Period must be a string
    def getStocks(tickers, period):
        dataframes = []
        for ticker_symbol in tickers:
            ticker = yf.Ticker(ticker_symbol)
            historical_data = ticker.history(period=period)

            if historical_data.isnull().any(axis=1).iloc[0]: #The first row can have Nan
                historical_data = historical_data.iloc[1:]

            #Assertions
            assert not historical_data.isnull().any(axis=None), f'history has NaNs in {ticker_symbol}'
            dataframes.append(historical_data)

        return pd.DataFrame(dataframes)
    def logReturns(price_data):
        close = price_data['Close'].values
        return np.log(close) - np.log(close.shift(1))

"""