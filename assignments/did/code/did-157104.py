import os
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# Cargar datos
data_path = os.path.join("..", "data", "callaway-santanna.csv")
df = pd.read_csv(data_path)

# Renombrar columnas
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

#  Variable de tiempo "k"
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df["k"] = df["t"] - df["treat_start"]

# Event dummies w/Nans
k_dummies = pd.get_dummies(df["k"], prefix="k", dummy_na=True, dtype=int)
df = pd.concat([df, k_dummies], axis=1)

# Indices
df = df.set_index(["i", "t"])

# ATT
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index
mask = mask_treated & df.index.get_level_values('i').isin(valid_units)
df_att = df[mask].copy()
all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]

# Periodo referencia
k_dummies = [k for k in all_k_dummies if k not in ['k_-1.0', 'k_nan']]

# Dummies
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

restriction0 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
values0 = np.array([0, 0, 0])  
f0=res0.wald_test(restriction0, values0)

anticipation0 = False
att0 = '-'

# ATT - All units
vars1 = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
# Definir dep1 e indep 
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

# Estimar modelo
res1 = m1.fit(cov_type="clustered")

# Probar efectos anticipados
restriction1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
values1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction1, values1)

# Anticipation 1 effect
anticipation1 = False
# Signo git aAt1
att1 = '-'