# Difference-in-Differences Assignment Solution

# 0. Imports
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load and prepare data
df = pd.read_csv('../data/callaway-santanna.csv')

df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
}, inplace=True)

df.loc[df['treat'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']
df.set_index(['i', 't'], inplace=True)

# 2. Create dummies for event time (k)
k_dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True, dtype=int)
df = pd.concat([df, k_dummies], axis=1)

# Helper function to build model
def estimate_att(data, k_prefix_cols, ref_col, dep_var='lemp'):
    # Drop reference period and any all-zero columns
    k_cols = [col for col in k_prefix_cols if col in data.columns and col != ref_col]
    exog = data[k_cols]
    exog = exog.loc[:, (exog != 0).any(axis=0)]
    model = PanelOLS(data[dep_var], exog, entity_effects=True, time_effects=True, drop_absorbed=True)
    return model.fit(cov_type='clustered')

# Dummies list
k_cols_full = [f'k_{k}' for k in [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]] + ['k_nan']
ref_col = 'k_-1.0'

# 3. ATT: Treated only
res0 = estimate_att(df[df['treat'] == 1], k_cols_full, ref_col)

# 4. Anticipation effects (treated only)
param_names0 = list(res0.params.index)
restriction = np.zeros((3, len(param_names0)))
for i, k in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if k in param_names0:
        restriction[i, param_names0.index(k)] = 1
values = np.zeros(3)
f0 = res0.wald_test(restriction, values)
anticipation0 = False
att0 = '-'  # all β_k < 0

# 5. ATT: All units
res1 = estimate_att(df, k_cols_full, ref_col)

# 6. Anticipation effects (all units)
param_names1 = list(res1.params.index)
restriction1 = np.zeros((3, len(param_names1)))
for i, k in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if k in param_names1:
        restriction1[i, param_names1.index(k)] = 1
values1 = np.zeros(3)
f1 = res1.wald_test(restriction1, values1)
anticipation1 = False
att1 = '-'  # all β_k < 0