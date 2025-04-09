# Imports
import os
import pandas as pd
import itertools
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Read data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)
df = df[df['yob'] > 1939]

# Crear dummies para 'qob'
dummies_qob = pd.get_dummies(df['qob'], prefix='qob', dtype=int)
# Crear dummies para 'yob'
dummies_yob = pd.get_dummies(df['yob'], prefix='yob', dtype=int)
# Agregar las dummies al DataFrame original
df = pd.concat([df, dummies_yob, dummies_qob], axis=1)

# Obtener listas de los años y trimestres
years = range(1940, 1950)  # Años de 1940 a 1949
quarters = range(1, 5)     # Trimestres del 1 al 4
combinations = itertools.product(years, quarters)

# Iterar y crear las columnas de interacción
for year, quarter in combinations:
    interaction_col_name = f'yob_{year}_qob_{quarter}'
    
    # Verificar que ambas columnas existen antes de realizar la multiplicación
    if f'yob_{year}' in df.columns and f'qob_{quarter}' in df.columns:
        df[interaction_col_name] = df[f'yob_{year}'] * df[f'qob_{quarter}']
    else:
        print(f"Columnas faltantes: yob_{year} o qob_{quarter}")

# Definir la variable dependiente
y = df['lwklywge']

# Crear las dummies para 'yob'
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', dtype=int)
yob_dummies = yob_dummies.drop(columns=['yob_1949'], errors='ignore')

# Definir las variables independientes (constante y controles)
X = df[['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent', 
        'soatl', 'esocent', 'wsocent', 'mt', 'educ']]

# Combinar los controles con las dummies del año de nacimiento
X = pd.concat([X, yob_dummies], axis=1)

# Agregar la constante al DataFrame de variables independientes
X = sm.add_constant(X)

# Ajustar el modelo ingenuo usando OLS (Regresión Lineal Ordinaria)
model = sm.OLS(y, X).fit(cov_type='HC3')  # HC3 para errores estándar robustos

# Almacenar los resultados en 'res0'
res0 = model.summary()

# Crear la lista de interacciones como instrumentos, omitiendo el último trimestre de cada año
instrument_cols = []
for year in years:
    for quarter in quarters:
        if not (quarter == 4):  # Omitir el último trimestre como referencia
            instrument_cols.append(f'yob_{year}_qob_{quarter}')

# Crear la matriz de instrumentos
instruments = df[instrument_cols]

# Verificar que las columnas de instrumentos existen
print(f"Instrumentos seleccionados: {instruments.columns.tolist()}")

# Redefinir las variables exógenas excluyendo 'educ' (que es endógena)
X_iv = df[['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent',
           'soatl', 'esocent', 'wsocent', 'mt']]

# Agregar las dummies del año de nacimiento, excluyendo 'yob_1949' como referencia
X_iv = pd.concat([X_iv, yob_dummies], axis=1)

# Agregar la constante
X_iv = sm.add_constant(X_iv)

# Definir la variable endógena (educ)
endog = df['educ']

# Ajustar el modelo 2SLS
iv_model = IV2SLS(dependent=y, exog=X_iv, endog=endog, instruments=instruments).fit()

# Almacenar los resultados en 'res1'
res1 = iv_model.summary

bias = True

bias_sign = '+'