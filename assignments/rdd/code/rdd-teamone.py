import pandas as pd
import statsmodels.api as sm
df = pd.read_csv('assignments/rdd/data/raw.csv')
THR = 40
df['x'] = df['cohsize'] - THR
spec1 = sm.OLS.from_formula(
    formula='avgverb ~ 1 + x + z + z:x',
    data=df
)
res = spec1.fit(cov_type='HC3')
