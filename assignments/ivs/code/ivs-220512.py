#Importing libraries
import os
import pandas as pd
import statsmodels.api as sm
import itertools
from linearmodels import IV2SLS

# 1.Loading Data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# 2. Filter out respondants born before 1940 (keep 1940 onward)
df = df.loc[df['yob'] >= 1940].copy()

# 3.Create a dummy for year of birth and quarter of birth (use `pd.get_dummies` to make your life easier).

# ---------- Birth Year ----------
yob_dummies = pd.get_dummies(df['yob'], prefix='yob', dtype=int)

# ---------- Birth Quarter ----------
qob_dummies = pd.get_dummies(df['qob'], prefix='qob', dtype=int)

# ---------- Cast to integer type ----------
yob_dummies = yob_dummies.astype(int)
qob_dummies = qob_dummies.astype(int)

# ---------- Adding Dummies to dataframe ----------
df = pd.concat([df, yob_dummies, qob_dummies], axis=1)

# 4. Create an interaction term for every year and every quarter using the dummies you created in the previous step.
# ---------- Dummies names ----------
years   = [c for c in df.columns if c.startswith('yob_')]
quarters = [c for c in df.columns if c.startswith('qob_')]

# ---------- Creating interactions ----------
for y, q in itertools.product(years, quarters):
    new_col = f'{y}_{q}'
    df[new_col] = df[y] * df[q]

# ---------- Keeping only qob_1, qob_2, qob_3 ----------
instr_cols = [c for c in df.columns if '_qob_4' not in c and '_qob_' in c]

# ---------- Controls ----------
base_controls = [
    'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
    'wnocent', 'soatl', 'esocent', 'wsocent', 'mt'
]

# ---------- Year Dummies 1949 is removed to get the reference group ----------
yob_controls = [c for c in years if c != 'yob_1949']


# ---------- Constant ----------
df['const'] = 1

# ---------- Regressors (Naive Model) ----------
X_ols = df[['const', 'educ'] + base_controls + yob_controls]

# ---------- Regressors (IV Model) ----------
X_iv  = df[['const'] + base_controls + yob_controls]

# ---------- Dependent Variable ----------
y = df['lwklywge']

# ---------- Endogenous ----------
endog = df['educ']

# ---------- Intruments ----------
Z = df[instr_cols]

# 5. Naive Model
res0 = sm.OLS(y, X_ols).fit(cov_type='HC3')

# 6. IV Model (2SLS)
res1 = IV2SLS(dependent=y,
                exog=X_iv,
                endog=endog,
                instruments=Z
               ).fit(cov_type='robust')

# 7. Does this evidence suggest that the naive model is biased?
beta_ols = res0.params['educ']
beta_iv  = res1.params['educ']
bias = True if (beta_ols > beta_iv or beta_ols < beta_iv) else False

# 8. Do you think the bias is negative, positive or zero (unbiased)?
bias_sign = '-' if bias < 0 else ('+' if bias > 0 else '0')