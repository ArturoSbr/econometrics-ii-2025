#imports
import os
import pandas as pd
import statsmodels.api as sm

# Load data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

## Renombrar columnas
df.columns = ['id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location']

# Mapear columnas a valores numéricos
df.replace(
    to_replace={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Mobile': '1', 'Desktop': '0'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'}
    },
    inplace=True
)

# Convertir strings a enteros
df[['dark', 'mobile', 'purchase']] = df[['dark', 'mobile', 'purchase']].astype(int)

# Convertir la columna 'location' a minúsculas
df['location'] = df['location'].str.lower()

df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

# Asignar una constante
df['const'] = 1

# Declarar modelo sin término de interacción
spec = sm.OLS(
    endog=df['purchase'],  # Variable dependiente
    exog=df[['const', 'ireland', 'scotland', 'wales', 'dark']],  # Variables explicativas sin interacción
)

# Ajustar el modelo
model = spec.fit()