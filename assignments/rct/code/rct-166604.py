# Imports
import os 
import pandas as pd
import statsmodels.api as sm

# Load data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Claean data

# Rename columns
df.columns = ['id', 'dark', 'views',
              'time', 'purchase', 'mobile',
              'location']

# Map columns to numeric dtypes
df.replace(
    to_replace ={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Desktop': '0', 'Mobile': '1'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'},
    },
    inplace = True
)

# Convert string into ints
df[['dark', 'mobile', 'purchase']] = df[['dark', 'mobile', 'purchase']].astype(int)

# OCD - lowercase
df['location'] = df['location'].str.lower()

#One-hot encoding 
df = pd.get_dummies(
    data = df,
    prefix = '',
    prefix_sep = '',
    columns = ['location'],
    dtype = int
)

# Feature engineering
df['cons'] = 1

# Declare linear model
spec = sm.OLS(
    endog = df['purchase'],
    exog = df[['cons', 'dark', 
               'ireland', 'scotland', 'wales',]],
               hasconst = True
    )

# Fit model
model = spec.fit()

