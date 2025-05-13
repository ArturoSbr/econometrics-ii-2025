import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ruta_archivo = os.path.join('..', 'data', 'callaway-santanna.csv')
datos = pd.read_csv(ruta_archivo)

datos.rename(columns={
    "year": "anio",
    "countyreal": "id_region",
    "first.treat": "inicio_trat"
}, inplace=True)

datos.loc[datos['inicio_trat'] == 0, 'inicio_trat'] = np.nan
datos['k_evento'] = datos['anio'] - datos['inicio_trat']

datos.set_index(['id_region', 'anio'], inplace=True)

dummies_k = pd.get_dummies(datos['k_evento'], prefix='k', dummy_na=True, dtype=int)
datos = datos.join(dummies_k)

mascara_tratadas = datos['inicio_trat'].notna()
panel_completo = ~datos['lemp'].isna().groupby(level='id_region').any()
regiones_validas = panel_completo[panel_completo].index
mascara_final = mascara_tratadas & datos.index.get_level_values('id_region').isin(regiones_validas)
datos_tratadas = datos[mascara_final].copy()

dummies_usar_tratadas = [col for col in datos_tratadas.columns if col.startswith('k_') and col != 'k_-1.0']
dummies_usar_tratadas = [col for col in dummies_usar_tratadas if datos_tratadas[col].nunique() > 1]

modelo_tratadas = PanelOLS(
    datos_tratadas['lemp'],
    datos_tratadas[dummies_usar_tratadas],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = modelo_tratadas.fit(cov_type='clustered')

R0 = np.identity(len(res0.params))[:3]
v0 = np.zeros((3, 1))
f0 = res0.wald_test(R0, v0)

anticipation0 = f0.pval < 0.05
att0 = '+' if res0.params.mean() > 0 else '-'

dummies_usar_todo = [col for col in datos.columns if col.startswith('k_') and col != 'k_-1.0' and datos[col].nunique() > 1]

modelo_todos = PanelOLS(
    datos['lemp'],
    datos[dummies_usar_todo],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res1 = modelo_todos.fit(cov_type='clustered')

R1 = np.identity(len(res1.params))[:3]
v1 = np.zeros((3, 1))
f1 = res1.wald_test(R1, v1)

anticipation1 = f1.pval < 0.05
att1 = '+' if res1.params.mean() > 0 else '-'
