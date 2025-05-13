import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Cargar datos
archivo = os.path.join('..', 'data', 'callaway-santanna.csv')
datos = pd.read_csv(archivo)

# Renombrar columnas
datos.rename(columns={
    'year': 'tiempo',
    'countyreal': 'unidad',
    'first.treat': 'trat_ini'
}, inplace=True)

# Crear variable evento
datos.loc[datos['trat_ini'] == 0, 'trat_ini'] = np.nan
datos['evento'] = datos['tiempo'] - datos['trat_ini']
datos.set_index(['unidad', 'tiempo'], inplace=True)

# Crear dummies de evento (sin kk_nan)
dummies_k = pd.get_dummies(datos['evento'], prefix='kk', dtype=int)
datos = datos.join(dummies_k)

# Filtrar unidades tratadas y con panel completo
tratadas = datos['trat_ini'].notna()
panel_completo = ~datos['lemp'].isna().groupby(level='unidad').any()
ids_validas = panel_completo[panel_completo].index
seleccion = tratadas & datos.index.get_level_values('unidad').isin(ids_validas)
datos_filtrados = datos[seleccion].copy()

# Modelo 1 (filtrado)
columnas_k = [c for c in datos_filtrados.columns if c.startswith('kk_') and c != 'kk_-1.0']
columnas_k = [c for c in columnas_k if datos_filtrados[c].nunique() > 1]
X1 = datos_filtrados[columnas_k]

modelo1 = PanelOLS(datos_filtrados['lemp'], X1, entity_effects=True, time_effects=True, drop_absorbed=True)
res0 = modelo1.fit(cov_type='clustered')

R0 = np.identity(len(X1.columns))[:3]
r0 = np.zeros(3)
f0 = res0.wald_test(R0, r0)

anticipacion0 = f0.pval < 0.05
efecto0 = '+' if res0.params.mean() > 0 else '-'

# Modelo 2 (datos sin filtrar, pero con limpieza de columnas válidas)
columnas_k_global = [c for c in datos.columns if c.startswith('kk_') and c != 'kk_-1.0']
columnas_k_global = [c for c in columnas_k_global if datos[c].nunique() > 1]
X2 = datos[columnas_k_global]

modelo2 = PanelOLS(datos['lemp'], X2, entity_effects=True, time_effects=True, drop_absorbed=True)
res1 = modelo2.fit(cov_type='clustered')

# Verificamos qué columnas están en el modelo realmente (algunas pueden ser absorbidas)
k_real = len(res1.params)
R1 = np.identity(k_real)[:3]
r1 = np.zeros(3)

f1 = res1.wald_test(R1, r1)
anticipacion1 = f1.pval < 0.05
efecto1 = '+' if res1.params.mean() > 0 else '-'
