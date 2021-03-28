from scipy.stats import norm

class VaR():
    def __init__(self, mean, stdv,confidence):
        self.confidence = confidence
        self.mean = mean
        self.stdv = stdv
    
    def valueAtRisk(self):
        return norm.ppf(1-self.confidence, self.mean,self.stdv)

    def simulateVaR(self, returns):
        return returns.quantile(1-self.confidence)

    def conditionalVaR(self):
        return (self.confidence**-1) * norm.pdf(norm.ppf(self.confidence))*self.stdv - self.mean