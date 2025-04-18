import os
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from itertools import product

# Read data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Filter respondents born from 1940 onwards
df = df[df['yob'] >= 1940]

# Create dummies for year of birth and quarter of birth
yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)

# Add dummies to the main DataFrame
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# Create interaction terms between yob and qob
interaction_terms = []
for year in range(1940, 1950):
    for quarter in range(1, 5):
        col_name = f'yob_{year}_qob_{quarter}'
        df[col_name] = df.get(f'yob_{year}', 0) * df.get(f'qob_{quarter}', 0)
        interaction_terms.append(col_name)

# Run naive OLS model
yob_dummies_naive = [f'yob_{y}' for y in range(1940, 1949)]
X_naive = df[['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
              'soatl', 'esocent', 'wsocent', 'mt', 'educ'] + yob_dummies_naive]
X_naive = sm.add_constant(X_naive)
y = df['lwklywge']
res0 = sm.OLS(y, X_naive).fit(cov_type='HC3')

# 2SLS IV model
exog = sm.add_constant(df[['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
                           'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'] + yob_dummies_naive])
filtered_terms = [col for col in interaction_terms if not col.endswith('qob_4')]
instruments = df[filtered_terms]
res1 = IV2SLS(dependent=df['lwklywge'],
              exog=exog,
              endog=df[['educ']],
              instruments=instruments).fit(cov_type='robust')

# Determine direction of bias
if res0.params['educ'] != res1.params['educ']:
    bias = 'True'
else:
    bias = 'False'    
if res0.params['educ'] > res1.params['educ']:
    bias_sign = '+'
elif res0.params['educ'] < res1.params['educ']:
    bias_sign = '-'
else:
    bias_sign = '0'
