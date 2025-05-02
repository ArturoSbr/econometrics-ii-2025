import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import itertools
from linearmodels import IV2SLS
print(os.getcwd())

# Q1 Cargar información y crear el data frame

df = pd.read_csv('../data/raw.csv')

# Q2 Filtramos para quedarnos con los que nacieron después de 1940

df = df[df['yob'] >= 1940]

#Q3 Creamos dummies para años

df = pd.get_dummies(
    data = df,
    columns=['yob','qob'],
    dtype = int
)

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

#Añadimos la constante que no estaba en la base
df['const'] = 1

#Definimos los controles con todas las dummies de año
X1 = ['const','race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
            'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'] + [col for col in df.columns if 'yob' in col and 'qob' not in col and '1949' not in col]

#Definimos los instrumentos que son las interacciones
I1 = [
    col for col in df.columns if 'yob' in col and 'qob' in col and 'qob_4' not in col
]

#Especificamos el modelo
m1 = IV2SLS(
    dependent = df['lwklywge'],
    exog = df[X1],
    endog = df['educ'],
    instruments=df[I1]
)

#Ajustamos el modelo
res1 = m1.fit()

# Q7 Sesgo

bias = 'True'
# Existe sesgo debido al problema de variables omitidas.

# Q8 Signo del sesgo

bias_sign = '+'

# Sin variables instrumentales se está sobreestimando el efecto