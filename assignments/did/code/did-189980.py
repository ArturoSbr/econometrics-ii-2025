import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Data
data_path = os.path.join("..", "data", "callaway-santanna.csv")
df = pd.read_csv(data_path)


# Rename columns
df.rename(columns={'year': 't', 'countyreal': 'i', 'first.treat': 'treat_start'}, inplace=True)

# New column k
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']

# Multi-index
df.set_index(['i', 't'], inplace=True)

# Dummies for event periods
dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True, dtype=int)
df = pd.concat([df, dummies], axis=1)

# ATT (treated)
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index
mask = mask_treated & df.index.get_level_values('i').isin(valid_units)
df_att = df[mask].copy()
all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]
k_dummies = [k for k in all_k_dummies if k not in ['k_-1.0', 'k_nan']]
k_dummies_final = [k for k in k_dummies if df_att[k].nunique() > 1]
exog0 = df_att[k_dummies_final]
m0 = PanelOLS(
    df_att["lemp"],
    exog0,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = m0.fit(cov_type="clustered")


# Wald Test
restriction0 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
values0 = np.array([0, 0, 0])  
f0=res0.wald_test(restriction0, values0)

# Anticipation effect
anticipation0 = False

# Effect of increasing wage
att0 = '-'

# ATT (all units)
event_cols_all = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
X_all = df[event_cols_all]
y_all = df['lemp']

model1 = PanelOLS(y_all, X_all, entity_effects=True, time_effects=True, drop_absorbed=True)
res1 = model1.fit(cov_type='clustered', cluster_entity=True)

# Wald Test all
restriction1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
values1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction1, values1)

# Anticipation effect
anticipation1 = False

# Effect of increasing wage
att1 = '-'
