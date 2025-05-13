import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

archivo = os.path.join('..', 'data', 'callaway-santanna.csv')
datos = pd.read_csv(archivo)

datos.rename(columns={
    'year': 'tiempo',
    'countyreal': 'unidad',
    'first.treat': 'trat_ini'
}, inplace=True)

datos.loc[datos['trat_ini'] == 0, 'trat_ini'] = np.nan
datos['evento'] = datos['tiempo'] - datos['trat_ini']
datos.set_index(['unidad', 'tiempo'], inplace=True)

dummies_k = pd.get_dummies(datos['evento'], prefix='kk', dummy_na=True, dtype=int)
datos = datos.join(dummies_k)

tratadas = datos['trat_ini'].notna()
panel_completo = ~datos['lemp'].isna().groupby(level='unidad').any()
ids_validas = panel_completo[panel_completo].index
seleccion = tratadas & datos.index.get_level_values('unidad').isin(ids_validas)
datos_filtrados = datos[seleccion].copy()

columnas_k = [c for c in datos_filtrados.columns if c.startswith('kk_') and c != 'kk_-1.0']
columnas_k = [c for c in columnas_k if datos_filtrados[c].nunique() > 1]
X1 = datos_filtrados[columnas_k]

modelo1 = PanelOLS(datos_filtrados['lemp'], X1, entity_effects=True, time_effects=True, drop_absorbed=True)
res0 = modelo1.fit(cov_type='clustered')

R0 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])
r0 = np.array([0, 0, 0])
f0 = res0.wald_test(R0, r0)

anticipacion0 = f0.pval < 0.05
efecto0 = '+' if res0.params.mean() > 0 else '-'

X2 = [c for c in datos.columns if c.startswith('kk_') and c != 'kk_-1.0' and datos[c].nunique() > 1]
modelo2 = PanelOLS(datos['lemp'], datos[X2], entity_effects=True, time_effects=True, drop_absorbed=True)
res1 = modelo2.fit(cov_type='clustered')

R1 = np.identity(len(X2))[:3]
r1 = np.zeros(3)
f1 = res1.wald_test(R1, r1)

anticipacion1 = f1.pval < 0.05
efecto1 = '+' if res1.params.mean() > 0 else '-'
