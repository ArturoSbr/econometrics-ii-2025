import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
from linearmodels import IV2SLS
#DATA
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

#FILTER
df = df[df['yob'] >= 1940]
yob_dummies = pd.get_dummies(df['yob'], prefix='yob')
qob_dummies = pd.get_dummies(df['qob'], prefix='qob')
yob_dummies = yob_dummies.astype(int)
qob_dummies = qob_dummies.astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)
df['const'] = 1

year_cols = [col for col in df.columns if col.startswith('yob_')]
quarter_cols = [col for col in df.columns if col.startswith('qob_')]

for year_col, quarter_col in itertools.product(year_cols, quarter_cols):
    name = f"{year_col}_{quarter_col}"  
    df[name] = df[[year_col, quarter_col]].prod(axis=1) 

#OLS
formula = 'lwklywge ~ const + race + married + smsa + neweng + midatl + ' \
          'enocent + wnocent + soatl + esocent + wsocent + mt + educ + ' \
          'yob_1940 + yob_1941 + yob_1942 + yob_1943 + yob_1944 + yob_1945 + yob_1946 + yob_1947 + yob_1948'

spec = sm.OLS.from_formula(formula, data=df)
res0 = spec.fit(cov_type='HC3')

#IV2SLS 
formula_iv = (
    'lwklywge ~ const + race + married + smsa + neweng + midatl + enocent + '
    'wnocent + soatl + esocent + wsocent + mt + '
    'yob_1940 + yob_1941 + yob_1942 + yob_1943 + yob_1944 + '
    'yob_1945 + yob_1946 + yob_1947 + yob_1948 + '
    '[educ ~ '
    'yob_1940_qob_1 + yob_1940_qob_2 + yob_1940_qob_3 + '
    'yob_1941_qob_1 + yob_1941_qob_2 + yob_1941_qob_3 + '
    'yob_1942_qob_1 + yob_1942_qob_2 + yob_1942_qob_3 + '
    'yob_1943_qob_1 + yob_1943_qob_2 + yob_1943_qob_3 + '
    'yob_1944_qob_1 + yob_1944_qob_2 + yob_1944_qob_3 + '
    'yob_1945_qob_1 + yob_1945_qob_2 + yob_1945_qob_3 + '
    'yob_1946_qob_1 + yob_1946_qob_2 + yob_1946_qob_3 + '
    'yob_1947_qob_1 + yob_1947_qob_2 + yob_1947_qob_3 + '
    'yob_1948_qob_1 + yob_1948_qob_2 + yob_1948_qob_3 + '
    'yob_1949_qob_1 + yob_1949_qob_2 + yob_1949_qob_3]'
)
spec2 = IV2SLS.from_formula(formula_iv, data=df)
res1 = spec2.fit()
 
bias = True
bias_sign = '+'