# Importar bibliotecas necesarias
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
# 1. Leer data 
PATH = os.path.join("..", "data", "callaway-santanna.csv")
df = pd.read_csv(PATH)
print(df.head())

# 2. Renombrar columnas clave
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

# 3. Ajustar valores para condados no tratados
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan

# 4. Crear la columna de tiempo relativo k = t - treat_start
df['k'] = df['t'] - df['treat_start']

# 5. Establecer el índice como panel (condado y año)
df = df.set_index(["i", "t"])

# 6. Crear variables dummy para cada período de evento
dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)
df = pd.concat([df, dummies], axis=1)

# 7. Mask_treated: ever-treated counties
mask_treated = ~df['treat_start'].isna()
# 8. Using the Mask_treated, filter the dataframe
df_t= df.loc[mask_treated]
# 9. identify the columns needed (k_-1 is the reference Value)
event_cols = [c for c in df_t.columns 
              if c.startswith('k_') and c not in ('k_-1.0', 'k_nan')]

# Ajustar el modelo para condados tratados
model_treated = PanelOLS(
    dependent=df_t['lemp'],
    exog=df_t[event_cols],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = model_treated.fit(cov_type='clustered', cluster_entity=True)

# 8. Prueba de efectos anticipados para tratados
params = res0.params.index.tolist()
R_matrix = np.zeros((3, len(params)))
for i, period in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if period in params:
        R_matrix[i, params.index(period)] = 1
v_vector = np.zeros((3, 1))
f0 = res0.wald_test(R_matrix, v_vector)

# 9. Declarar resultados de tratados
anticipation0 = False
att0 = '+'

# 10. Estimar ATT para todos los condados
event_vars_all = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']

model_all = PanelOLS(
    dependent=df['lemp'],
    exog=df[event_vars_all],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
# fit model
res1 = model_all.fit(cov_type='clustered', cluster_entity=True)

# 11. Prueba de anticipación para todos
params_all = res1.params.index.tolist()
R_all = np.zeros((3, len(params_all)))
for i, period in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    if period in params_all:
        R_all[i, params_all.index(period)] = 1
v_all = np.zeros((3, 1))
f1 = res1.wald_test(R_all, v_all)

# 12. Declarar resultados para todos
anticipation1 = False
att1 = '-'
