# %% [markdown]
# # Randomized Controlled Trials
# 
# We'll be taking a look at an online retailer based in the United Kingdom. Our
# goal is to estimate the causal effect of switching the user's interface to dark
# on the probability of purchasing an item.
# 
# We will fit the following model:
# 
# $$ E(Y_i | X_i) = X_i^T \gamma + \tau D_i$$
# 
# where $X_i$ are controls and $D_i$ indicates $i$'s treatment status.
# 
# ---
# 
# ## Imports

# %%
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# %% [markdown]
# ## Exploratory Data Analysis
# 
# Load data

# %%
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# %% [markdown]
# View data

# %%
df.head()

# %% [markdown]
# Check for nulls

# %%
df.isna().sum()

# %% [markdown]
# Check data types

# %%
df.dtypes

# %% [markdown]
# Clean data

# %%
# Rename columns
df.columns = ['id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location']

# Map columns to numeric dtypes
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

# Set `location`` to lowercase
df['location'] = df['location'].str.lower()

# %% [markdown]
# Encode categorical variables to binary columns (also known as One-Hot Encoding)

# %%
df.head()

# %%
# One-hot encoding
df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)

# %% [markdown]
# Feature engineering
# 
# - Create interaction term
# - Assign a constant

# %%
# Interaction
df['dark_mobile'] = df['dark'].multiply(df['mobile'])

# Constant
df['const'] = 1

# %% [markdown]
# ## Fitting a linear model
# 
# Declare linear model

# %%
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

# %% [markdown]
# - Interaction term is not significant (remove it)

# %% [markdown]
# Fitting a parsimonious model

# %%
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


