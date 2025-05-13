import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Step 1: Load Data
PATH = os.path.join('..', 'data', 'callaway-santanna.csv')
df = pd.read_csv(PATH)

# Step 2: Rename Columns
df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
}, inplace=True)

# Step 3: Set never-treated units to NaN and create time-to-treatment k
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']

# Step 4: Set multi-index
df.set_index(['i', 't'], inplace=True)

# Step 5: Create dummies for event time
k_dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True, dtype=int)
df = df.join(k_dummies)

# Step 6: ATT (only treated units)
treated_mask = df['treat_start'].notna()
panel_mask = ~df['lemp'].isna().groupby(level='i').any()
valid_ids = panel_mask[panel_mask].index
final_mask = treated_mask & df.index.get_level_values('i').isin(valid_ids)
df_att = df[final_mask].copy()

# Use all k dummies except the reference category
all_k = [col for col in df_att.columns if col.startswith('k_')]
k_vars0 = [k for k in all_k if k != 'k_-1.0' and df_att[k].nunique() > 1]

# Step 7: Estimate ATT for treated
spec0 = PanelOLS(
    df_att['lemp'],
    df_att[k_vars0],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = spec0.fit(cov_type='clustered')

# Step 8: Wald Test for anticipation effects
restriction0 = np.identity(len(k_vars0))[:3]
values0 = np.zeros(3)
f0 = res0.wald_test(restriction0, value=values0)

anticipation0 = f0.pval < 0.05
att0 = '+' if res0.params.mean() > 0 else '-'

# Step 9: ATT for all units
k_vars1 = [k for k in all_k if k != 'k_-1.0' and df[k].nunique() > 1]

spec1 = PanelOLS(
    df['lemp'],
    df[k_vars1],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res1 = spec1.fit(cov_type='clustered')

# Step 10: Wald Test for anticipation effects (all units)
restriction1 = np.identity(len(k_vars1))[:3]
values1 = np.zeros(3)
f1 = res1.wald_test(restriction1, value=values1)

anticipation1 = f1.pval < 0.05
att1 = '+' if res1.params.mean() > 0 else '-'
