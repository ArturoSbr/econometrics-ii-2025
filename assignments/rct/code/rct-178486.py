##assigment RCT Ariadna Martinez 178486
#Imports 
import os
import numpy as np 
import pandas as pd  
import statsmodels.api as sm 

#load data 
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

#clean data
df.columns = ['id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location']

# Map columns to numeric values and spelling changes 
df.replace(
    to_replace={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Mobile': '1', 'Desktop': '0'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'}
    },
    inplace=True
)

# strings to int 
df[['dark', 'mobile', 'purchase']] = df[['dark', 'mobile', 'purchase']].astype(int)

# Convert  'location' column 
df['location'] = df['location'].str.lower()

#one-hot encoding 
df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

#constant 
df['const'] = 1

#fitting model 

spec2 = sm.OLS(
    endog=df['purchase'],  # Variable dependiente
    exog=df[['const', 'ireland', 'scotland', 'wales', 'dark']],  # Variables explicativas sin interacción
)

# Ajustar el modelo
model = spec2.fit()

# Ver resultados
model.summary()
