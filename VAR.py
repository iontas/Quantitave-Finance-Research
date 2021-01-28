def value_at_risk(returns, confidence_level=.05):
    return returns.quantile(confidence_level, axis=0, interpolation='higher')

def expected_shortfall(returns, confidence_level=.05):
    var = value_at_risk(returns, confidence_level)
    #ES is the average of worst losses
    return returns[returns.lt(var, axis=1)].mean()
