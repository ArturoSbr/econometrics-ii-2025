#Imports 
import os
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

#1. Load data
path= os.path.join('..','data', 'callaway-santanna.csv')
df = pd.read_csv(path)

#2. Rename columns
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

#3. Declare new time column k
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df["k"] = df["t"] - df["treat_start"]

#4. Set multi-index
df = df.set_index(["i", "t"])

#5. Create dummies for event periods
k_dummies = pd.get_dummies(df["k"], prefix="k", dummy_na=True, dtype=int)
df = pd.concat([df, k_dummies], axis=1)

#6. ATT (only treated counties)
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index

mask = mask_treated & df.index.get_level_values('i').isin(valid_units)

df_att = df[mask].copy()

all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]


k_dummies = [k for k in all_k_dummies if k not in ['k_-1.0', 'k_nan']]


k_dummies_final = [k for k in k_dummies if df_att[k].nunique() > 1]

indep = df_att[k_dummies_final]

spec0 = PanelOLS(
    df_att["lemp"],
    indep,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = spec0.fit(cov_type="clustered")

#7. Check for anticipated effects
params = res0.params.index.tolist()
R_matrix = np.zeros((3, len(params)))
for i, period in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if period in params:
        R_matrix[i, params.index(period)] = 1
v_vector = np.zeros((3, 1))
f0 = res0.wald_test(R_matrix, v_vector)

#8. Is there evidence of anticipation effects?
anticipation0 = False
# Aticipation effect sign 
att0 = '-'

#9. ATT all units
event_vars_all = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']

#Estimate model
model_all = PanelOLS(
    dependent=df['lemp'],
    exog=df[event_vars_all],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
#Fitted model
res1 = model_all.fit(cov_type='clustered', cluster_entity=True)

#11. Check for anticipated effects
params_all = res1.params.index.tolist()
R_all = np.zeros((3, len(params_all)))
for i, period in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if period in params_all:
        R_all[i, params_all.index(period)] = 1
v_all = np.zeros((3, 1))
f1 = res1.wald_test(R_all, v_all)

#12. Is there evidence of anticipation?
anticipation1 = False

#13. Is the effect of increasing the minimum wage positive or negative?
att1 = '-'



