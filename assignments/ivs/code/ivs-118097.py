#Imports
import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# 1. Load the data in an object named "df"

PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)


# 2. Filter out respondents born before 1940
df = df[df['yob'] >= 1940]

#3. Create dummies for year of birth and quarter of birth
# Create year dummies for 1940-1949
year_dummies = pd.get_dummies(df['yob'], prefix='yob')
year_dummies = year_dummies.loc[:, [col for col in year_dummies.columns if int(col.split('_')[1]) <= 1949]]

# Convert boolean to integer
year_dummies = year_dummies.astype(int)

# Create quarter dummies
quarter_dummies = pd.get_dummies(df['qob'], prefix='qob')

# Convert boolean to integer
quarter_dummies = quarter_dummies.astype(int)

# Add the dummies to the original dataframe
df = pd.concat([df, year_dummies, quarter_dummies], axis=1)

# 4. Create interaction terms for every year and every quarter
# Get lists of year and quarter columns
year_cols = [col for col in year_dummies.columns]
quarter_cols = [col for col in quarter_dummies.columns]

# Create dictionary for new interaction columns
interaction_dict = {}

# Use itertools to create all combinations
for year_col, quarter_col in itertools.product(year_cols, quarter_cols):
    year = year_col.split('_')[1]
    quarter = quarter_col.split('_')[1]
    new_col_name = f"yob_{year}_qob_{quarter}"
    interaction_dict[new_col_name] = df[year_col] * df[quarter_col]

# Add the interaction terms to the dataframe
df = df.assign(**interaction_dict)

# Verify the result
print(f"DataFrame shape: {df.shape}")
print("\nSample of year dummies:")
print(df[[col for col in df.columns if col.startswith('yob_') and '_qob_' not in col]].head())
print("\nSample of quarter dummies:")
print(df[[col for col in df.columns if col.startswith('qob_')]].head())
print("\nSample of interaction terms:")
print(df[[col for col in df.columns if '_qob_' in col]].iloc[:, :5].head())

# 5 Run a Naive Model

# Add a constant to the dataframe
df['const'] = 1

# 1. Run naive model (OLS) with robust standard errors
# Define the control variables:
# We exclude yob_1949 to use it as reference
controls = ['const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent', 'wnocent', 
            'soatl', 'esocent', 'wsocent', 'mt', 'educ']

# Add year of birth dummies (excluding 1949 as reference)
yob_dummies = [col for col in df.columns if col.startswith('yob_') and '_qob_' not in col 
               and col != 'yob_1949']
all_controls = controls + yob_dummies

# Run OLS regression
X = df[all_controls]
y = df['lwklywge']

model = sm.OLS(y, X)
res0 = model.fit(cov_type='HC3')  # Using robust standard errors (HC3)

print("Naive Model Results:")
print(res0.summary())

# 2. Run IV2SLS model
# Prepare instruments: interactions between yob and qob
# For each year, we'll exclude the 4th quarter as reference


# Define exogenous variables (same as controls but without educ)
exog_vars = ['const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
            'wnocent', 'soatl', 'esocent', 'wsocent', 'mt']
exog_vars.extend([f'yob_{year}' for year in range(1940, 1949)])
instruments = [f'yob_{year}_qob_{quarter}' 
                   for year in range(1940, 1950)
                   for quarter in range(1, 4)]

# Run IV2SLS regression
formula = {"dependent": df['lwklywge'], 
           "exog": df[exog_vars], 
           "endog": df[['educ']], 
           "instruments": df[instruments]}

iv_model = IV2SLS(**formula)
res1 = iv_model.fit(cov_type='robust')

print("\nIV2SLS Model Results:")
print(res1.summary)

# 3 & 4. Compare coefficients to determine bias
# Get the coefficient of education from both models
coef_naive = res0.params['educ']
coef_iv = res1.params['educ']

print(f"\nCoefficient of education in naive model: {coef_naive:.4f}")
print(f"Coefficient of education in IV model: {coef_iv:.4f}")
print(f"Difference (IV - naive): {coef_iv - coef_naive:.4f}")

bias = True
bias_sign = "+"