import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

archivo_datos = os.path.join('..', 'data', 'callaway-santanna.csv')
datos_panel = pd.read_csv(archivo_datos)

datos_panel.rename(columns={
    'year': 'anio',
    'countyreal': 'region_id',
    'first.treat': 'inicio_tratamiento'
}, inplace=True)

datos_panel.loc[datos_panel['inicio_tratamiento'] == 0, 'inicio_tratamiento'] = np.nan
datos_panel['periodo_evento'] = datos_panel['anio'] - datos_panel['inicio_tratamiento']
datos_panel.set_index(['region_id', 'anio'], inplace=True)

dummies_evento = pd.get_dummies(datos_panel['periodo_evento'], prefix='evento', dummy_na=True, dtype=int)
datos_panel = datos_panel.join(dummies_evento)

es_tratada = datos_panel['inicio_tratamiento'].notna()
panel_completo = ~datos_panel['lemp'].isna().groupby(level='region_id').any()
regiones_validas = panel_completo[panel_completo].index
filtrado = es_tratada & datos_panel.index.get_level_values('region_id').isin(regiones_validas)
datos_tratadas = datos_panel[filtrado].copy()

columnas_evento = [col for col in datos_tratadas.columns if col.startswith('evento_') and col != 'evento_-1.0']
columnas_evento = [col for col in columnas_evento if datos_tratadas[col].nunique() > 1]
X_treat = datos_tratadas[columnas_evento]

modelo_tratadas = PanelOLS(datos_tratadas['lemp'], X_treat, entity_effects=True, time_effects=True, drop_absorbed=True)
resultados_tratadas = modelo_tratadas.fit(cov_type='clustered')

restriccion_treat = np.identity(len(resultados_tratadas.params))[:3]
valores_nulos = np.zeros((3, 1))
test_f0 = resultados_tratadas.wald_test(restriccion_treat, valores_nulos)

anticipacion0 = test_f0.pval < 0.05
efecto_estimado0 = '+' if resultados_tratadas.params.mean() > 0 else '-'

columnas_evento_full = [col for col in datos_panel.columns if col.startswith('evento_') and col != 'evento_-1.0' and datos_panel[col].nunique() > 1]
modelo_completo = PanelOLS(datos_panel['lemp'], datos_panel[columnas_evento_full], entity_effects=True, time_effects=True, drop_absorbed=True)
resultados_completos = modelo_completo.fit(cov_type='clustered')

restriccion_all = np.identity(len(resultados_completos.params))[:3]
valores_nulos_all = np.zeros((3, 1))
test_f1 = resultados_completos.wald_test(restriccion_all, valores_nulos_all)

anticipacion1 = test_f1.pval < 0.05
efecto_estimado1 = '+' if resultados_completos.params.mean() > 0 else '-'
