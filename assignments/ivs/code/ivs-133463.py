# Import libraries
import pandas as pd
import os
import itertools
import statsmodels.api as sm
from linearmodels import IV2SLS

# Data loading
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Filter pre-1940 observations
df = df.query('yob >= 1940')

# Dummy creation
df = pd.get_dummies(data=df, dtype='int', columns=['yob', 'qob'])

# Interaction terms
Y_COLS = [c for c in df.columns if c.startswith('yob')]
Q_COLS = [c for c in df.columns if c.startswith('qob')]
INTERACTIONS = {f'{y}_{q}': df[y]*df[q] for y, q in itertools.product(Y_COLS, Q_COLS)}
df = df.assign(**INTERACTIONS)

# Variables for formula
DEP_VAR = 'lwklywge'
NON_YOB_CONTROLS = (
    'race + married + smsa + neweng + '
    'midatl + enocent + wnocent + soatl + '
    'esocent + wsocent + mt'
)
YOB_CONTROLS = ' + '.join([c for c in Y_COLS if '1949' not in c])
ENDOG_VAR = 'educ'
INTERACTION_TERMS = [
    c for c in df.columns if
    c.startswith('yob') and
    not c.endswith('qob_4') and
    not c.endswith('yob_1949') and
    c not in YOB_CONTROLS
]
INSTRUMENTS = ' + '.join([c for c in INTERACTION_TERMS])

# Formulas
FORMULA_RES0 = f'{DEP_VAR} ~ 1 + {NON_YOB_CONTROLS} + {YOB_CONTROLS} + {ENDOG_VAR}'
FORMULA_RES1 = f'{DEP_VAR} ~ 1 + {NON_YOB_CONTROLS} + {YOB_CONTROLS} + [{ENDOG_VAR} ~ {INSTRUMENTS}]'
print(FORMULA_RES0)
print(FORMULA_RES1)

# Model specs
spec0 = sm.OLS.from_formula(formula=FORMULA_RES0, data=df)
spec1 = IV2SLS.from_formula(formula=FORMULA_RES1, data=df)

# Estimation
res0 = spec0.fit(cov_type='HC3')
res1 = spec1.fit()
print(res0.summary())
print(res1)

# Bias analysis
bias = True
bias_sing = '+'
