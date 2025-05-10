# assignments/ivs/code/ivs-<your id here>.py

import pandas as pd
import itertools
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

# Load data
df = pd.read_csv('../data/raw.csv')

# Filter respondents born from 1940 onward
df = df[df['yob'] >= 1940].copy()

# Create dummy variables for year and quarter of birth
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', dtype=int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob', dtype=int)

# Join dummies to dataframe
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# Create interaction terms using safer assignment
interaction_terms = {}
for yob, qob in itertools.product(range(1940, 1950), range(1, 5)):
    yob_col = f'yob_{yob}'
    qob_col = f'qob_{qob}'
    if yob_col in df.columns and qob_col in df.columns:
        interaction_terms[f'{yob_col}_{qob_col}'] = df[yob_col] * df[qob_col]

# Add interaction terms to the DataFrame
df = df.assign(**interaction_terms)

# Add constant term
df['const'] = 1

# Naive OLS Model
controls_naive = [
    'const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt', 'educ'
] + [f'yob_{yr}' for yr in range(1940, 1949)]  # Reference: 1949

res0 = sm.OLS(df['lwklywge'], df[controls_naive]).fit(cov_type='HC3')

# IV Model
interaction_cols = [f'yob_{y}_qob_{q}' for y in range(1940, 1950) for q in range(1, 4)]  # omit qob_4 each year

controls_iv = [
    'const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt'
] + [f'yob_{yr}' for yr in range(1940, 1949)]  # Reference: 1949

res1 = IV2SLS(
    dependent=df['lwklywge'],
    exog=df[controls_iv],
    endog=df['educ'],
    instruments=df[interaction_cols]
).fit(cov_type='robust')

# Check for bias
bias = True
bias_sign = '+'
