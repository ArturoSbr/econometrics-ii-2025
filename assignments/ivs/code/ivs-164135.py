import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
df = pd.read_csv('../data/raw.csv')
df= df[df['cohort'] == '40-49']
year_dummies = pd.get_dummies(df['yob'], prefix='yob', drop_first=False)
year_dummies = year_dummies.astype(int)
quarter_dummies = pd.get_dummies(df['qob'], prefix='qob', drop_first=False)
quarter_dummies = quarter_dummies.astype(int)
df = pd.concat([df, year_dummies, quarter_dummies], axis=1)
years = [f'yob_{year}' for year in range(1940, 1950)]  
quarters = [f'qob_{qtr}' for qtr in range(1, 5)]       
interaction_terms = {}
for year, quarter in itertools.product(years, quarters):
    interaction_col_name = f'{year}_{quarter}'
    interaction_terms[interaction_col_name] = df[year] * df[quarter]
df = df.assign(**interaction_terms)

X = df[['educ', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent', 
        'soatl', 'esocent', 'wsocent', 'mt', 'race'] + 
        [col for col in df.columns if col.startswith('yob_') and '1949' not in col and 'qob_'not in col]]
X = sm.add_constant(X)
y = df['lwklywge']
model = sm.OLS(y, X)
res0 = model.fit(cov_type='HC3')

from linearmodels import IV2SLS

controls =['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent', 
            'soatl', 'esocent', 'wsocent', 'mt']+[col for col in df.columns if col.startswith('yob_') and '1949' not in col and 'qob_'not in col]

instruments = [
    col for col in df.columns
    if any(f"yob_{año}_qob_" in col for año in range(1940, 1949)) and not col.endswith("qob_4")
]
exog = '1 + ' + ' + '.join(controls)
instr = ' + '.join(instruments)
formula = f"lwklywge ~ {exog} + [educ ~ {instr}]"
model = IV2SLS.from_formula(formula, data=df)
res1 = model.fit(cov_type='robust')
bias = True
bias_sign = '+'