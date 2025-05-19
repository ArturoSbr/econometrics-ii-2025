#Importar
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

#1
PATH = os.path.join('..', 'data', 'callaway-santanna.csv')
df = pd.read_csv(PATH)

#2
df.rename(columns={'year': 't', 'countyreal': 'i', 'first.treat':'treat_start'}, inplace=True)

#3
df.loc[df['treat'] == 0, 'treat_start'] = np.nan
df['k'] = df['t']-df['treat_start']

#4
df.set_index(['i','t'], inplace = True)

#5
dummies_k = pd.get_dummies(df['k'], prefix='k', dummy_na=True, dtype=int)
df = pd.concat([df, dummies_k], axis=1)

#6
mask1 = df['treat'] == 1
df_treated = df[mask1].copy()
exog = df_treated[[col for col in df.columns if col.startswith('k_') and col not in ['k_-1.0', 'k_nan']]]
endog = df_treated['lemp']
res0 = PanelOLS(
    endog,
    exog,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

#7
params = list(res0.params.index)
restr_vars = ['k_-4.0', 'k_-3.0', 'k_-2.0']
restr_vars_in_model = [v for v in restr_vars if v in params]
restr_vars_missing = [v for v in restr_vars if v not in params]
R = np.zeros((len(restr_vars_in_model), len(params)))
for i, var in enumerate(restr_vars_in_model):
    j = params.index(var)
    R[i, j] = 1

q = np.zeros(len(restr_vars_in_model))
f0 = res0.wald_test(R, q)

#8
anticipation0 = False

#9
att0 ='-'

#10
exog_all = df[[col for col in df.columns if col.startswith('k_') and col not in ['k_-1.0', 'k_nan']]]
endog_all = df['lemp']

res1 = PanelOLS(
    endog_all,
    exog_all,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

#11
params1 = list(res1.params.index)
restr_vars1 = ['k_-4.0', 'k_-3.0', 'k_-2.0']
restr_vars_in_model1 = [v for v in restr_vars1 if v in params1]

R1 = np.zeros((len(restr_vars_in_model1), len(params1)))
for i, var in enumerate(restr_vars_in_model1):
    j = params1.index(var)
    R1[i, j] = 1

q1 = np.zeros(len(restr_vars_in_model1))

f1 = res1.wald_test(R1, q1)

#12
anticipation1 = False

#13
att1 ='-'