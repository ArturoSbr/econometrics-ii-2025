import pandas as pd
import itertools
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

#1. Cargamos los datos
df = pd.read_csv('../data/raw.csv')

#2. Filtramos los datos, años mayores a 1940
df = df[df['yob'] >= 1940]

#3. Generamos dummies para año y quarter de nacimiento
yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

#4. Generamos interacciones entre año y quartes
yob_cols = [col for col in df.columns if col.startswith('yob_') and '_qob_' not in col]
qob_cols = [col for col in df.columns if col.startswith('qob_')]
for yob, qob in itertools.product(yob_cols, qob_cols):
    df[f'{yob}_{qob}'] = (df[yob] * df[qob]).astype(int)

#5. Agregamos la constante
df['const'] = 1

#5. Modelo MCO 
controls = [
    'educ', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies = [col for col in df.columns if col.startswith('yob_') and '1949' not in col and 'qob_' not in col]
X = df[['const'] + controls + yob_dummies]
y = df['lwklywge']
res0 = sm.OLS(y, X).fit(cov_type='HC3')

#6. Modelo IV
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

#7. Respuesta sesgo
bias = True
#8. Signo sesgo
bias_sign = '+'