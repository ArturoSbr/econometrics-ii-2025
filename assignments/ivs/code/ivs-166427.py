import os
import pandas as pd
from itertools import product
from linearmodels import IV2SLS
import statsmodels.api as sm

df = pd.read_csv("../data/raw.csv")

df = df.loc[df.loc[:, "yob"] >= 1940]

dummies_yob = pd.get_dummies(df['yob'], prefix='yob').astype(int)
dummies_qob = pd.get_dummies(df['qob'], prefix='qob').astype(int)

df = pd.concat([df, dummies_yob, dummies_qob], axis=1)

year_cols = ["yob_1940","yob_1941","yob_1942","yob_1943","yob_1944","yob_1945","yob_1946","yob_1947","yob_1948","yob_1949"]
quarter_cols = ["qob_1","qob_2","qob_3","qob_4"]

interactions = []
for col in year_cols:
    for col2 in quarter_cols:
        new_col_name = col + "_" + col2
        interactions.append(new_col_name)
        df[new_col_name] = df[col] * df[col2]

control_vars = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt', 'educ'
]
year_dummies = [col for col in df.columns if col.startswith('yob_') and not col.startswith('yob_1949')]
X = df[control_vars + year_dummies]
X = sm.add_constant(X)
y = df['lwklywge']
model = sm.OLS(y, X)
res0 = model.fit(cov_type='HC3')

yob_qob_to_drop = [col for col in df.columns if col.endswith("qob_4")]
df = df.drop(columns=yob_qob_to_drop)

controls = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt'
]

year_dummies = [col for col in df.columns if col.startswith('yob_') and not col.startswith('yob_1949') and not col in interactions]
instrument_dummies = [col for col in df.columns if not col.endswith("qob_4") and col in interactions]
exog = sm.add_constant(df[controls + year_dummies])
endog = df['educ']
instruments = df[instrument_dummies]
dep = df['lwklywge']
res1 = IV2SLS(dependent=dep, exog=exog, endog=endog, instruments=instruments).fit(cov_type='robust')

bias = True

bias_sign = '+'
