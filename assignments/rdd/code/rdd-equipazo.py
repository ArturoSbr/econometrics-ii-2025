# Librerías
import pandas as pd
import os
import statsmodels.api as sm

# Carga de datos
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# Variable X
CUTOFF = 40
df['x'] = df['cohsize'] - CUTOFF

# Estimación de modelo
spec = sm.OLS.from_formula(formula='avgverb ~ 1 + x + z + x*z', data=df)
res = spec.fit(cov_type='HC3')
