# Imports
import os
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS 

# Load data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Data validation
df.isna().sum()
df.dtypes

cohort = '40-49'
df_filtered = df[df['cohort'] == cohort]

# Create dummies for yob and qob using pd.get_dummies
yob_dummies = pd.get_dummies(df_filtered['yob'], prefix = 'yob').astype(int)
qob_dummies = pd.get_dummies(df_filtered['qob'], prefix = 'qob').astype(int)


# merge dummies with df_filtered
df_filtered = pd.concat([df_filtered, yob_dummies, qob_dummies], axis = 1)

# 1. Create interaction terms
yob_cols = [f'yob_{year}' for year in range(1940, 1950)]
qob_cols = [f'qob_{q}' for q in range(1, 5)]

# 2. Create iteractions
interaction_terms = {
    f'{yob}_{qob}': df_filtered[yob] * df_filtered[qob]
    for yob, qob in itertools.product(yob_cols, qob_cols)
}

# 3. Assign interction terms to df_filtered
df_filtered = df_filtered.assign(**interaction_terms) 

# Add a constant term 
df_filtered['const'] = 1

# Create control variables
control_vars = [
    'const', 'race', 'married', 'smsa', 'neweng', 'midatl',
    'enocent', 'wnocent', 'soatl', 'esocent', 'wsocent', 'mt', 'educ'
] + [f'yob_{year}' for year in range(1940, 1949)] 

# Naive model
model = sm.OLS(df_filtered['lwklywge'], df_filtered[control_vars])
res0 = model.fit(cov_type='HC3')

# 1. Prepare the variables
df_filtered['const'] = 1  # Add constant term

# 2. Define control variables (1949 as reference for year dummies)
controls = [
    'const', 'race', 'married', 'smsa', 'neweng', 'midatl',
    'enocent', 'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
] + [f'yob_{year}' for year in range(1940, 1949)]

# 3. Prepare instruments (interactions between yob and qob, excluding qob_4 for each year)
instruments = []
for year in range(1940, 1949):
    for quarter in range(1, 4): 
        instruments.append(f'yob_{year}_qob_{quarter}')

# 4. Verify full column rank of instruments
X_inst = df_filtered[instruments]
if np.linalg.matrix_rank(X_inst.values) < X_inst.shape[1]:
    raise ValueError("Instruments don't have full column rank")

# 5. Correct formula specification for IV2SLS
# The proper format is: 'depvar ~ exog_vars + [endog ~ instruments]'
formula = (
    'lwklywge ~ const + race + married + smsa + neweng + midatl + '
    'enocent + wnocent + soatl + esocent + wsocent + mt + '
    + ' + '.join([f'yob_{year}' for year in range(1940, 1949)])
    + ' + [educ ~ ' + ' + '.join(instruments) + ']'
)

# IV model
res1 = IV2SLS.from_formula(formula, data=df_filtered).fit(cov_type='robust')

bias = True  
bias_sign = '+' 
