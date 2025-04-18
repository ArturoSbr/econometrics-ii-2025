import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import itertools
from linearmodels.iv import IV2SLS
print(os.getcwd())

# Q1 Cargar información y crear el data frame
df = pd.read_csv('../data/raw.csv')

# Q2 Filtramos para quedarnos con los que nacieron después de 1940
df = df[df['yob'] >= 1940]

#Q3 Creamos dummies para años
yob_d = pd.get_dummies(df['yob'], prefix='yob', drop_first=False)
yob_d = yob_d.astype(int)

#Creamos dummies para quarters
qrt_d = pd.get_dummies(df['qob'], prefix='qob', drop_first=False)
qrt_d = qrt_d.astype(int)

#Pegamos al dataframe base
df = pd.concat([df, yob_d, qrt_d], axis=1)

# Q4 Creamos interacciones

# Lista de columnas de años y trimestres
year_cols = [col for col in df.columns if col.startswith('yob_')]
qob_cols = [col for col in df.columns if col.startswith('qob_')]

# Crear todas las interacciones posibles con los dos vectores anteriores
interactions = {
    f'{y}_{q}': df[y] * df[q]
    for y, q in itertools.product(year_cols, qob_cols)
}

# Pegarlos al dataframe
df = df.assign(**interactions)

# Q5 - Naive model

# Definimos los controles que vamos a usar
controls = ['educ','race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 
            'wnocent', 'soatl', 'esocent', 'wsocent', 'mt']
# Nos quedamos sólo con las dummies de años
year_dummies = [col for col in df.columns if col.startswith('yob_') and col != 'yob_1949' and 'qob' not in col]

# Hacemos la matriz de controles con constantes y definimos la variable independiente
X = sm.add_constant(df[controls + year_dummies])
y = df['lwklywge']

#Definimos el modelo
model = sm.OLS(y, X)

#Corremos el modelo con errores HC3
res0 = model.fit(cov_type='HC3')

#Q6 IV model

#Definimos los controles otra vez sin las interacciones
controls = ['race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
            'wnocent', 'soatl', 'esocent', 'wsocent', 'mt']
year_dummies = [col for col in df.columns if col.startswith('yob_') and col != 'yob_1949'and 'qob' not in col]
exog = sm.add_constant(df[controls + year_dummies])

#Definimos la variable endógena y la dependiente
endog = df['educ']
dependent = df['lwklywge']

#Definimos el instrumento que son las interacciones, excepto el q4
instrument_cols = [col for col in df.columns 
                   if any(y in col for y in year_dummies) and col.startswith('yob_') 
                   and not col.endswith('_qob_4')]
instruments = df[instrument_cols]

# Generamos los vectores con los que se va a correr el modelo eliminando si es que hubiera nas
data_iv = pd.concat([dependent, exog, endog, instruments], axis=1).dropna()
dependent = data_iv[dependent.name]
exog = data_iv[exog.columns]
endog = data_iv[endog.name]
instruments = data_iv[instruments.columns]

# Corremos el modelo
res1 = IV2SLS(dependent, exog, endog, instruments).fit(cov_type='robust')

# Q7 Sesgo

bias = 'True'
# Existe sesgo debido al problema de variables omitidas.

# Q8 Signo del sesgo

bias_sign = '+'

# Sin variables instrumentales se está sobreestimando el efecto