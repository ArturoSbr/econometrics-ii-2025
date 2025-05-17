##IMPORTS
import os
import numpy as np
import pandas as pd
from linearmodels import PanelOLS  # For panel data regression

###LOAD DATA
did_PATH = os.path.join('..', 'data', "callaway-santanna.csv")
df = pd.read_csv(did_PATH)

###Rename Columns
df = df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
})

### Declare new time column k
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']

###Set multi-index, 

df = df.set_index(['i', 't'])

# Create event period dummies from k
# First convert k to categorical with NaN handling
k_dummies = pd.get_dummies(df['k'].astype('category'), 
                          prefix='k', 
                          dummy_na=True, 
                          dtype='int')
# Clean up column names (removes .0 for integer periods)
k_dummies.columns = k_dummies.columns.str.replace('.0', '', regex=False)
# Join dummies back to main dataframe
df = pd.concat([df, k_dummies], axis=1)


##ATT (only treated counties)
# Crear una máscara para unidades tratadas
mask_tratadas = df['treat_start'].notna()
# Filtrar unidades con panel completo (sin valores faltantes en la variable dependiente)
panel_completo = ~df['lemp'].isna().groupby(level='i').any()
unidades_validas = panel_completo[panel_completo].index
# Crear una máscara final que combine ambas condiciones
filtro_final = mask_tratadas & df.index.get_level_values('i').isin(unidades_validas)
datos_filtrados = df[filtro_final].copy()
# Seleccionar variables de tiempo relativo al tratamiento (dummies)
dummies_tiempo = [col for col in datos_filtrados.columns if col.startswith('k_')]
dummies_utiles = [col for col in dummies_tiempo if col != 'k_-1.0']
dummies_finales = [col for col in dummies_utiles if datos_filtrados[col].nunique() > 1]
variables_exogenas = datos_filtrados[dummies_finales]
# Estimar modelo de efectos fijos con efectos de entidad y tiempo
modelo = PanelOLS(
    datos_filtrados['lemp'],
    variables_exogenas,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

res0=modelo.fit(cov_type='clustered')

###Check for anticipated effects WALD TEST
n_coefs = len(res0.params)
restriction = np.zeros((3, n_coefs))  # 3 restricciones x N coeficientes

# Asignar 1's para los coeficientes a testear (β_-4, β_-3, β_-2)
coef_names = res0.params.index
for i, k in enumerate([-4, -3, -2]):
    col_idx = list(coef_names).index(f'k_{k}')
    restriction[i, col_idx] = 1

values = np.array([0, 0, 0])  # H0: β_-4 = β_-3 = β_-2 = 0

# Realizar prueba de Wald
f0 = res0.wald_test(restriction, values)

#Evidence of anticipation Effects?
anticipation0= False
#Is the effect of increasing the minimum wage positive or negative?
att0= '-'

# ATT (All units)

vars_exog_att = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
matriz_exog_att = df[vars_exog_att]

modelo_att = PanelOLS(
    df['lemp'],
    matriz_exog_att,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

res1 = modelo_att.fit(cov_type='clustered')

#Check for anticipated effects
#Run a Wald test
# Asumir orden de coeficientes: [k_-4, k_-3, k_-2, k_0, k_1, k_2, k_3]
n_coefs1 = len(res1.params)
restriction = np.zeros((3, n_coefs1))  # 3 restricciones x 7 coeficientes

restriction[0, 0] = 1  # β_{-4} = 0
restriction[1, 1] = 1  # β_{-3} = 0
restriction[2, 2] = 1  # β_{-2} = 0
values = np.array([0, 0, 0])  # H0: Rβ = [0,0,0]^T

# Ejecutar prueba de Wald
f1 = res1.wald_test(restriction, values)

#Evidence of anticipation Effects?
anticipation1= False
#Is the effect of increasing the minimum wage positive or negative?
att1= '-'






