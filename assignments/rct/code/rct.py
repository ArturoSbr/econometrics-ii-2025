# Imports
import os
import pandas as pd
import statsmodels.api as sm

# Load data
PATH_DATA = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH_DATA)

# Rename columns
df.columns = [
    'id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location'
]

# Map treatment and device columns
df.replace(
    to_replace={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Desktop': '0', 'Mobile': '1'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'},
    },
    inplace=True
)

# Convert strings to integers
df[['dark', 'mobile', 'purchase']] = df[
    ['dark', 'mobile', 'purchase']
].astype(int)

# Lowercase location
df['location'] = df['location'].str.lower()

# Enconde location
df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

# Create interaction term
df['dark_mobile'] = df['dark'].multiply(df['mobile'])

# Add constant
df['const'] = 1

# # Specify model
spec = sm.OLS(
    endog=df['purchase'],
    exog=df[[
        'const', 'ireland', 'scotland', 'wales', 'dark'
    ]],
    hasconst=True
)

# Fit model
model = spec.fit()
