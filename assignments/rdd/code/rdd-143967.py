import os
import pandas as pd
import statsmodels.api as sm

# Read data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Fit polynomial model
# Feature engineering
CUTOFF = 40
df['x'] = df['cohsize'] - CUTOFF

# Declare and fit model
spec = sm.OLS.from_formula(formula='avgverb ~ 1 + x + + z + z * x', data=df)
res = spec.fit(cov_type='HC3')
