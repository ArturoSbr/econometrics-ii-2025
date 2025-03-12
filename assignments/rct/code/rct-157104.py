# Imports
import os
import pandas as pd
import statsmodels.api as sm

# Load data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Rename columns
df.columns = ['id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location']

# Map columns to numeric types
df.replace(
    to_replace={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Mobile': '1', 'Desktop': '0'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'}
    },
    inplace=True
)

# Convert strings -> ints
df[['dark', 'mobile', 'purchase']] = df[['dark', 'mobile', 'purchase']].astype(int)
# A minusculas
df['location'] = df['location'].str.lower()

# One-hot encoding
df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

# Interaction
df['dark_mobile'] = df['dark'].multiply(df['mobile'])

# Constant
df['const'] = 1

# Declare specification
spec = sm.OLS(
    endog=df['purchase'],
    exog=df[['const', 'ireland', 'scotland', 'wales', 'dark', 'dark_mobile']],
    hasconst=True
)

# Fit model
model = spec.fit()

# View results
model.summary()

# Declare model
spec2 = sm.OLS(
    endog=df['purchase'],
    exog=df[['const', 'ireland', 'scotland', 'wales', 'dark']],  # No interaction
    hasconst=True
)

# Fit model
model2 = spec2.fit()

# View results
model2.summary()
