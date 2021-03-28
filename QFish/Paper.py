from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
"""
Gamma Distribution
Γ(α,β) to denote the gamma distribution
shape parameterα >0 and rate parameterβ >0.
writeU([0,1]) todenote the uniform distribution in [0,1] andP(λ) to denote the Poisson distributionwith parameterλ >0.

The probability density function for gamma is:
f(x,a)=x^(a-)e^(-x)/Γ(a)
"""
"""
matplot lib figure
"""
fig, ax = plt.subplots(1, 1)
a = 1.99 #shape parameter
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')

#Plot PDF
x = np.linspace(gamma.ppf(0.01,a), gamma.ppf(0.99, a), 100)
ax.plot(x, gamma.pdf(x, a),'r-',lw=2,alpha=0.6, label='gamma pdf')
ax.legend(loc='best', frameon=False)

plt.show()
"""
-Gamma Distribution
-Poisson Distribution
-Uniform Distribution
"""
