# Loading Libraries
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
PATH = os.path.join('..','data','raw.csv')
df=pd.read_csv(PATH)

# Rename Columns
df.columns=['id', 'dark', 'view','time','purchase','mobile','location']

# Map columns to numeric type
df.replace(
    to_replace={
        'dark':{'A':'0','B':'1'},
        'mobile':{'Mobile':'1', 'Desktop':'0'},
        'purchase':{'No':'0','Yes':'1'},
        'location':{'Northern Ireland':'Ireland'}
        }
,inplace=True)

# Convert  strings -> ints
df[['dark','mobile','purchase']]=df[['dark','mobile','purchase']].astype(int)

# set location to lowercase
df['location']=df['location'].str.lower()

# one-hot encoding
df=pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

# Assign a contant
df['const']=1

# Declare Model
spec=sm.OLS(
    endog=df['purchase'],
    exog=df[['const','ireland','scotland','wales','dark']],
    hasconst=True
)

# Fit Model
model=spec.fit()

# View Results
model.params['dark']