# Importar librerias
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

#Q1 importar datos
df = pd.read_csv('../did/data/callaway-santanna.csv')

#Q2 Renombrar columnas
df = df.rename(columns = {'year': 't',
                          'countyreal': 'i',
                          'first.treat': 'treat_start'})

#Q3 Crear nueva columna k
df.loc[df['treat'] == 0, 'treat_start'] = np.nan
df['k'] = df['t']-df['treat_start']

#Q4 Establecer índices, primero i luego t
df.set_index(['i','t'], inplace = True)

#Q5 Crear dummies para periodos antes y después del evento
df = pd.get_dummies(data=df, columns=['k'],prefix='k', dummy_na=True ,dtype=int, drop_first=False)

#Q6 ATT sólo para tratados

#Crear mascara para sólo los tratados
mask1 = df['treat'] == 1

#Identificar observaciones con datos completos - sin NaN en lemp que será nuestra variable dependiente
completos1 = df.loc[mask1].groupby('i')['lemp'].apply(lambda x: not x.isna().any()).loc[lambda x: x].index

#Aplicamos los dos filtros pasados
df_completo = df.loc[df.index.get_level_values('i').isin(completos1) & mask1].copy()

#Identificamos todas las dummies que usaremos, es decir, las k antes y después del evento
dummies1 = [col for col in df_completo.columns if col.startswith('k_')]

#Escluimos la de referencia que es k-1
dummies2 = [k for k in dummies1 if k != 'k_-1.0']

#De las anteriores las que tiene más de un valor, es decir 0 y 1
dummies3 = [k for k in dummies2 if df_completo[k].nunique() > 1]

#Con esto ya tenemos nuestros controles
X1 = df_completo[dummies3].copy()

#Definimos el modelo
m0 = PanelOLS(
    df_completo['lemp'],  # Solo tratados con panel completo
    X1,                      # Matriz de variables explicativas (dummies k)
    entity_effects=True,        # Efectos fijos por condado
    time_effects=True,          # Efectos fijos por año
    drop_absorbed=True          # Eliminar variables colineales automáticamente
)

#Ajustamos el modelo
res0 = m0.fit(cov_type='clustered', cluster_entity=True)

#Q7 Anticipo en el tratamiento si alguna B antes del tratamiento es distinta de cero

mat1 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
h0 = np.array([0, 0, 0])  
f0=res0.wald_test(mat1, h0)

#Q8 No hay efectos anticipados
anticipation0 = False

#Q9 El efecto es negativo
att0 ='-'

#Q10 ATT para todos

#Identificar observaciones con datos completos - sin NaN en lemp que será nuestra variable dependiente
completos1 = df.groupby('i')['lemp'].apply(lambda x: not x.isna().any()).loc[lambda x: x].index

#Aplicamos el filtro de datos completos sin usar mask1
df_completo = df.loc[df.index.get_level_values('i').isin(completos1)].copy()

#Identificamos todas las dummies que usaremos, es decir, las k antes y después del evento
dummies1 = [col for col in df_completo.columns if col.startswith('k_')]

#Excluimos la de referencia que es k-1
dummies2 = [k for k in dummies1 if k != 'k_-1.0']

#De las anteriores las que tienen más de un valor, es decir, 0 y 1
dummies3 = [k for k in dummies2 if df_completo[k].nunique() > 1]

#Con esto ya tenemos nuestros controles
X1 = df_completo[dummies3].copy()

#Definimos el modelo
m1 = PanelOLS(
    df_completo['lemp'],  # Todos los condados (tratados y no tratados)
    X1,                    # Matriz de variables explicativas (dummies k)
    entity_effects=True,   # Efectos fijos por condado
    time_effects=True,     # Efectos fijos por año
    drop_absorbed=True     # Eliminar variables colineales automáticamente
)

#Ajustamos el modelo
res1 = m1.fit(cov_type='clustered', cluster_entity=True)

#Q11 Anticipo en el tratamiento si alguna B antes del tratamiento es distinta de cero

mat2 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
h1 = np.array([0, 0, 0])  
f1=res1.wald_test(mat2, h1)

#Q12 No hay efectos anticipados
anticipation1 = False

#Q13 El efecto es negativo
att1 ='-'
