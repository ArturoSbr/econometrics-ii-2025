

import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

current_dir = os.getcwd()
relative_path = os.path.join(current_dir, '..', 'data', 'callaway-santanna.csv')
absolute_path = os.path.abspath(relative_path)
df = pd.read_csv(absolute_path)

df = df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
})

df['treat_start'] = df['treat_start'].replace(0, np.nan)
df['k'] = df['t'] - df['treat_start']

df = df.set_index(['i', 't'])

k_dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)
df = pd.concat([df, k_dummies], axis=1)

treated_df = df[df['treat_start'].notna()].copy()
k_cols = [col for col in treated_df.columns if col.startswith('k_')]
if 'k_-1.0' in k_cols:
    k_cols.remove('k_-1.0')
X = treated_df[k_cols]
X = X.loc[:, X.nunique() > 1]
y = treated_df['lemp']
res0 = PanelOLS(
    dependent=y,
    exog=X,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

restriction = np.array([
    [1, 0, 0, 0, 0, 0],  # β_{-4}
    [0, 1, 0, 0, 0, 0],  # β_{-3}
    [0, 0, 1, 0, 0, 0]   # β_{-2}
])
values = np.array([[0], [0], [0]])
f0 = res0.wald_test(restriction, value=values)

anticipation0 = False

att0 = '+'

k_cols = [col for col in df.columns if col.startswith('k_')]
if 'k_-1.0' in k_cols:
    k_cols.remove('k_-1.0')
X = df[k_cols]
X = X.loc[:, X.nunique() > 1]
y = df['lemp']
res1 = PanelOLS(
    dependent=y,
    exog=X,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

restriction = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # β_{-4} = 0
    [0, 1, 0, 0, 0, 0, 0],  # β_{-3} = 0
    [0, 0, 1, 0, 0, 0, 0]   # β_{-2} = 0
])

values = np.array([[0], [0], [0]])
f1 = res1.wald_test(restriction, value=values)

anticipation1 = False

att1 = '-'