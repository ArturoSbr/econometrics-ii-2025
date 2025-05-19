#importar librerias
import os
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# 1. Cargar datos
data_path = os.path.join("..", "data", "callaway-santanna.csv")
df = pd.read_csv(data_path)

# 2. Renombrar columnas
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

#  3. Declare time variable k 
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df["k"] = df["t"] - df["treat_start"]

# 4. Crear dummies de eventos, incluyendo NaNs
k_dummies = pd.get_dummies(df["k"], prefix="k", dummy_na=True, dtype=int)
df = pd.concat([df, k_dummies], axis=1)

# 5. Establecer índice múltiple
df = df.set_index(["i", "t"])

# 6. Efecto solo en Tratados
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index

mask = mask_treated & df.index.get_level_values('i').isin(valid_units)

df_att = df[mask].copy()

all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]

# Excluir referencia y k_nan
k_dummies = [k for k in all_k_dummies if k not in ['k_-1.0', 'k_nan']]

# Mantener solo dummies que tienen variación
k_dummies_final = [k for k in k_dummies if df_att[k].nunique() > 1]

exog0 = df_att[k_dummies_final]

spec0 = PanelOLS(
    df_att["lemp"],
    exog0,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

res0 = spec0.fit(cov_type="clustered")

# 7. Compruebe los efectos previstos
restriction0 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
values0 = np.array([0, 0, 0])  
f0=res0.wald_test(restriction0, values0)

# 8. ¿Existe evidencia de efectos de anticipación?
anticipation0 = False
# 9. ¿El efecto del aumento del salario mínimo es positivo o negativo?
att0 = '-'

# 10. ATT (todas las unidades)
vars1 = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
# Paso 2: definir dependiente e independientes
y1 = df["lemp"]
X1 = df[vars1]

# Paso 3: definir modelo
m1 = PanelOLS(
    dependent=y1,
    exog=X1,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

# Paso 4: estimar modelo (¡usar m1!)
res1 = m1.fit(cov_type="clustered")

# 11. Compruebe los efectos previstos (todas las unidades)
restriction1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
values1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction1, values1)

# 12. ¿Existe evidencia de efectos de anticipación?
anticipation1 = False
# 13. ¿El efecto del aumento del salario mínimo es positivo o negativo?
att1 = '-'
