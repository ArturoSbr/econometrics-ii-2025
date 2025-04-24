import os
import pandas as pd
from itertools import product
from linearmodels import IV2SLS
import statsmodels.api as sm

# Cargar los datos
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Filtrar los respondientes nacidos después de 1940
df = df[df['yob'] >= 1940]

# Crear las variables dummy para yob y qob (quarter of birth)
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', drop_first=True).astype(int)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob', drop_first=True).astype(int)

# Añadir las dummies al dataframe
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# Crear términos de interacción yob * qob
interaction_terms = list(product(yob_dummies.columns, qob_dummies.columns))
for yob_col, qob_col in interaction_terms:
    df[f'{yob_col}_{qob_col}'] = df[yob_col] * df[qob_col]

# Variables de control
controls = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]

# Eliminar filas con NaN en las variables relevantes
df = df.dropna(subset=controls + ['lwklywge'])

# Añadir constante para regresión OLS
df['const'] = 1

# Redefinir X e y después de limpiar datos
X = df[['const'] + controls]
y = df['lwklywge']

# Crear modelo OLS sin ajustar
model_naive = sm.OLS(y, X)

# Ajustar modelo con errores robustos y guardar resultado como 'res0'
res0 = model_naive.fit(cov_type='HC3')
print(res0.summary())

# Variables para el modelo IV
endogenous = ['educ']
instruments = [col for col in df.columns if 'yob_' in col and 'qob_' in col and '_' in col]  # Interacciones yob*qob

# Validación opcional (evitar error si faltan columnas por algún filtro anterior)
for col in instruments:
    if col not in df.columns:
        print(f"Warning: Missing instrument column {col}")

# Ajustar modelo IV2SLS
model_iv = IV2SLS(dependent=y,
                  exog=X[controls],
                  endog=X[endogenous],
                  instruments=df[instruments]).fit()

res1 = model_iv.summary
print(res1)

# Evaluar si el modelo OLS está sesgado
bias = True  # Asumiendo que hay endogeneidad en 'educ'
bias_sign = '+'  # Si el coeficiente en OLS es mayor al del IV, hay sesgo positivo
