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

# Crear las variables dummy para yob y el trimestre de nacimiento
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', drop_first=True)
qob_dummies = pd.get_dummies(df['qob'], prefix='qob', drop_first=True)

# Asegurarse de que las variables sean enteros, no booleanos
yob_dummies = yob_dummies.astype(int)
qob_dummies = qob_dummies.astype(int)

# Añadir las dummies al dataframe
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# Crear los términos de interacción entre el año  el trimestre de nacimiento
interaction_terms = list(product(yob_dummies.columns, qob_dummies.columns))

# Bucle para crear las columnas de interacción
for yob_col, qob_col in interaction_terms:
    df[f'{yob_col}_{qob_col}'] = df[yob_col] * df[qob_col]


df['const'] = 1

controls = [
    'educ', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]
yob_dummies = [col for col in df.columns if col.startswith('yob_')
               and '1949' not in col and 'qob_' not in col]

X = df[['const'] + controls + yob_dummies]
y = df['lwklywge']

print(X.dtypes)
print(y.dtypes)

# Asegurarse de que X y y sean numéricas
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Eliminar filas con valores NaN (si es necesario)
df = df.dropna(subset=controls + ['lwklywge'])

# Añadir constante al dataframe
df['const'] = 1

# Crear la fórmula para el modelo
X = df[controls]
y = df['lwklywge']

# Ajustar el modelo lineal con errores estándar robustos (HC3)
model_naive = sm.OLS(y, X).fit(cov_type='HC3')

# Mostrar el resumen del modelo
res0 = model_naive.summary()

# Estimar el modelo de dos etapas (2SLS) usando IV2SLS
# Definir la variable endógena y los instrumentos
endogenous = ['educ']
instruments = [col for col in df.columns if 'yob' in col and 'qob' in col]

# Crear el modelo IV2SLS
model_iv = IV2SLS(y, X[controls], endog=X[endogenous], instrument=X[instruments]).fit()
res1 = model_iv.summary()


# Evaluar si el modelo ingenuo está sesgado
bias = None
bias_sign = None

# Aquí puedes insertar tu análisis sobre si el modelo ingenuo está sesgado o no
# (Por ejemplo, comparando los resultados de ambos modelos)

bias = True  # Ajusta esto según tu análisis
bias_sign = '+'  # Ajusta el signo del sesgo según tu análisis
