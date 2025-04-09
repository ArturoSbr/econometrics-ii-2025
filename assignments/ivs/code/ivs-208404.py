import pandas as pd
import statsmodels.api as sm
import itertools
import numpy as np
from linearmodels.iv import IV2SLS

df = pd.read_csv('../data/raw.csv')
df = df[df['yob'] >= 1940]
yob_dummies = pd.get_dummies(df['yob'], prefix='yob')
yob_dummies = yob_dummies.loc[:, [f'yob_{y}' for y in range(1940, 1950)]].astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob')
qob_dummies = qob_dummies.loc[:, [f'qob_{q}' for q in range(1, 5)]].astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)
interaction_cols = {
    f'{y}_{q}': df[y] * df[q]
    for y, q in itertools.product(yob_dummies.columns, qob_dummies.columns)
}
df = df.assign(**interaction_cols)
formula = (
    "lwklywge ~ educ + race + married + smsa + "
    "neweng + midatl + enocent + wnocent + soatl + "
    "esocent + wsocent + mt + C(yob, Treatment(1949))"
)
model = sm.OLS.from_formula(formula, data=df)
res0 = model.fit(cov_type='HC3')
controls = [
    'race', 'married', 'smsa',
    'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt'
] + [f'yob_{y}' for y in range(1940, 1949)]

instr_vars = [
    f'yob_{y}_qob_{q}'
    for y in range(1940, 1950)
    for q in range(1, 4)
    if f'yob_{y}_qob_{q}' in df.columns
]
endog = df['educ']
exog = df[controls]
instr = df[instr_vars]
res1 = IV2SLS(
    dependent=df['lwklywge'],
    exog=sm.add_constant(exog, has_constant='add'),
    endog=endog,
    instruments=sm.add_constant(instr, has_constant='add')
).fit(cov_type='robust')
bias = True
bias_sign = '+'