# Imports
import os
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

#1. Load data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

#2. Filter
df = df[df['yob'] >= 1940]

#3. Dummies
yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

#4. Interaction terms
interaction_terms = {}
for yob, qob in itertools.product(range(1940, 1950), range(1, 5)):
    yob_col = f'yob_{yob}'
    qob_col = f'qob_{qob}'
    if yob_col in df.columns and qob_col in df.columns:
        interaction_terms[f'{yob_col}_{qob_col}'] = df[yob_col] * df[qob_col]
df = df.assign(**interaction_terms)

#5. Naive model
df['const'] = 1
controls_naive = [
    'const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt', 'educ'
] + [f'yob_{yr}' for yr in range(1940, 1949)]  # Reference: 1949

res0 = sm.OLS(df['lwklywge'], df[controls_naive]).fit(cov_type='HC3')

#6. 2sls IV
controls_iv = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies_iv = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and '_qob_' not in col]
instr_cols = [col for col in df.columns if '_qob_' in col and not col.endswith('_qob_4')]

exog = df[['const'] + controls_iv + yob_dummies_iv]
endog = df['educ']
instruments = df[instr_cols]
res1 = IV2SLS(dependent=df['lwklywge'], exog=exog, endog=endog, instruments=instruments).fit(cov_type='robust')

#7. Bias
bias = True

#8. Sign of bias
bias_sign = '+'