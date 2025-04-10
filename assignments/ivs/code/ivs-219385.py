# Cargar librerías
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from itertools import product
import numpy as np

# Cargar y preparar datos
df = pd.read_csv('../data/raw.csv')
df = df[df['yob'] >= 1940]

# Crear dummies para año y trimestre
yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# Crear interacciones yob × qob (excepto _qob_4 se usará como referencia después)
yob_cols = [col for col in df.columns if col.startswith('yob_') and '_qob_' not in col]
qob_cols = [col for col in df.columns if col.startswith('qob_')]
for yob, qob in product(yob_cols, qob_cols):
    df[f'{yob}_{qob}'] = (df[yob] * df[qob]).astype(int)

# Agregar constante
df['const'] = 1

# Modelo OLS (res0) 
controls = [
    'educ', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and 'qob_' not in col]
X = df[['const'] + controls + yob_dummies]
y = df['lwklywge']
res0 = sm.OLS(y, X).fit(cov_type='HC3')

# Modelo IV (res1)
controls_iv = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies_iv = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and '_qob_' not in col]
instr_cols = [col for col in df.columns if '_qob_' in col and not col.endswith('_qob_4')]

exog = df[['const'] + controls_iv + yob_dummies_iv]
endog = df['educ']
instruments = df[instr_cols]
res1 = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instruments).fit(cov_type='robust')

# Evaluación del sesgo
bias = True
bias_sign = '+'

