# Imports
import os
import numpy as np
import pandas as pd
from itertools import product
import statsmodels.api as sm
from linearmodels.iv import IV2SLS 

df = pd.read_csv('..', 'data', 'raw.csv')
df = df[df['yob'] >= 1940]

yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

yob_cols = [col for col in df.columns if col.startswith('yob_') and '_qob_' not in col]
qob_cols = [col for col in df.columns if col.startswith('qob_')]
for yob, qob in product(yob_cols, qob_cols):
    df[f'{yob}_{qob}'] = (df[yob] * df[qob]).astype(int)

df['const'] = 1

controls = [
    'educ', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and 'qob_' not in col]

X = df[['const'] + controls + yob_dummies]
y = df['lwklywge']
res0 = sm.OLS(y, X).fit(cov_type='HC3')


controls_iv = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies_iv = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and '_qob_' not in col]
instr_cols = [col for col in df.columns if '_qob_' in col and not col.endswith('_qob_4')]

exog = df[['const'] + controls_iv + yob_dummies_iv]
endog = df['educ']
instruments = df[instr_cols]

res1 = IV2SLS(
    dependent=df['lwklywge'],
    exog=exog,
    endog=endog,
    instruments=instruments
).fit(cov_type='robust')

bias = True
bias_sign = '+' 
