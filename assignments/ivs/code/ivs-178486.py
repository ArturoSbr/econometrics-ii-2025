import pandas as pd
import numpy as np
from itertools import product
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
#cargar data 
df = pd.read_csv('assignments/ivs/data/raw.csv')
# Filtrar años de nacimiento entre 1940 y 1949
df = df[(df['yob'] >= 1940) & (df['yob'] <= 1949)].copy()
# Crear la columna cohort con el valor '40-49'
df['cohort'] = '40-49'
## Crear dummies
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', dtype=int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob', dtype=int)
# Agregar las dummies al DataFrame
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)
# Crear interacciones yob_{year}_qob_{quarter}
for y_col, q_col in product(yob_dummies.columns, qob_dummies.columns):
    year = y_col.split('_')[1]
    quarter = q_col.split('_')[1]
    interaction_name = f"yob_{year}_qob_{quarter}"
    df[interaction_name] = df[y_col] * df[q_col]
# Convertir todo a numérico
df = df.apply(pd.to_numeric, errors='coerce')

# Controles base
base_controls = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]

# Dummies de YOB (excluye yob_1949 como referencia)
yob_dummy_names = [col for col in yob_dummies.columns if col != "yob_1949"]

# Interacciones yob_qob (excluye *_qob_4 como referencia)
instr_cols = [
    col for col in df.columns
    if "_qob_" in col and not col.endswith("_qob_4")
]
# Constante
df["const"] = 1.0
#Definir matrices para regresión
X_ols = df[["const", "educ"] + base_controls + yob_dummy_names]
X_iv = df[["const"] + base_controls + yob_dummy_names]
Z = df[instr_cols].copy()
# Evitar columnas duplicadas en Z
Z = Z[[col for col in Z.columns if col not in X_iv.columns]]
# Variable dependiente y endógena
y = df["lwklywge"]
endog = df["educ"]
res0 = sm.OLS(y, X_ols).fit(cov_type='HC3')
# 7. IV Model (2SLS)
res1 = IV2SLS(dependent=y, exog=X_iv, endog=endog, instruments=Z).fit(cov_type='robust')

# Diagnóstico de sesgo
beta_ols = res0.params['educ']
beta_iv = res1.params['educ']
bias = True if (beta_ols != beta_iv) else False
bias_sign = '+' if beta_ols > beta_iv else ('-' if beta_ols < beta_iv else '0')