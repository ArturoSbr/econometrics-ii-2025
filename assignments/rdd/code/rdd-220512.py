import os
import pandas as pd
import statsmodels.api as sm

# Reading data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

#Seting the cutoff
CUTOFF = 40
df['x'] = df['cohsize'] - CUTOFF

# Declaring and fit model
spec = sm.OLS.from_formula(formula='avgverb ~ 1 + x + + z + z * x', data=df)
res = spec.fit(cov_type='HC3')

