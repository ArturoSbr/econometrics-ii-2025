"""
Difference-in-Differences Analysis: Effect of Minimum Wage on Teen Employment
Based on Callaway & Sant'Anna (2021)
"""

# 1. Importar librerías necesarias
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# 2. Cargar los datos
data_path = os.path.join('..','data', 'callaway-santanna.csv')
df = pd.read_csv(data_path)

# Column rename
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

#  Time variable "k"
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df["k"] = df["t"] - df["treat_start"]

# Event dummies w/Nans
k_dummies = pd.get_dummies(df["k"], prefix="k", dummy_na=True, dtype=int)
df = pd.concat([df, k_dummies], axis=1)

# Multiindex
df = df.set_index(["i", "t"])


# ATT
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index

mask = mask_treated & df.index.get_level_values('i').isin(valid_units)

df_att = df[mask].copy()

all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]

# Drop ref period
k_dummies = [k for k in all_k_dummies if k not in ['k_-1.0', 'k_nan']]

# Keep dummies 
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

# Check 4 anticipated effects
restriction0 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
values0 = np.array([0, 0, 0])  
f0=res0.wald_test(restriction0, values0)

# Anticipation effect?
anticipation0 = False
# Aticipation effect sign 
att0 = '-'

# ATT - All units
vars1 = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
# Define dep1 and indep 
dep1 = df["lemp"]
indep1 = df[vars1]

# 
m1 = PanelOLS(
    dependent=dep1,
    exog=indep1,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

# Estimate model
res1 = m1.fit(cov_type="clustered")

# Test anticipated effects
restriction1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
values1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction1, values1)

# Anticipation 1 effect
anticipation1 = False
# Effect sign At1
att1 = '-'