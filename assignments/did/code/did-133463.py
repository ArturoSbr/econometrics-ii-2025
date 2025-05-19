# Librerias
import os
import numpy as np
import pandas as pd
from linearmodels import PanelOLS

# Carga de datos
PATH = os.path.join('..', 'data', 'callaway-santanna.csv')
df = pd.read_csv(PATH)

# Renombrar columnas
df.rename(columns={'year':'t', 'countyreal':'i', 'first.treat':'treat_start'}, inplace=True)

# NA a "treat_start" si i en control
county_treatment_detail = (
    df
    .groupby(['i'], as_index=False)['treat']
    .apply(lambda x: 1 if x.sum() > 0 else 0)
)
df['treat_start'] = (
    np
    .select(
        [
            (df['i'].isin(county_treatment_detail[county_treatment_detail['treat']==0]['i']))
        ],
        [
            np.nan
        ],
        default=df['treat_start']
    )
)
df['k'] = df['t'] - df['treat_start']

# Multi-index
df.set_index(['i', 't'], inplace=True)

# Dummies
df = (
    pd
    .get_dummies(
        data=df, 
        columns=['k'], 
        prefix='k', 
        prefix_sep='_',
        dummy_na=True,
        dtype='int'
    )
)

# Modelo 0
treated_pop = (df['treat']==1)
m0 = (
    PanelOLS(
        dependent=df[treated_pop]['lemp'],
        exog=df[treated_pop][[x for x in df.columns if 'k' in x and 'k_nan' not in x and 'k_-1.0' not in x]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
)
res0 = m0.fit(cov_type='clustered')

# Test de Wald - Modelo 0
R0 = (
    np
    .array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ]
    )
)
V = (
    np
    .array(
        [0, 0, 0]
    )
)
f0 = res0.wald_test(R0, V)
anticipation0 = False
att0 = '-'

# Modelo 1
m1 = (
    PanelOLS(
        dependent=df['lemp'],
        exog=df[[x for x in df.columns if 'k' in x and 'k_nan' not in x and 'k_-1.0' not in x]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
)
res1 = m1.fit(cov_type='clustered')

# Test de Wald - Modelo 1
R1 = (
    np
    .array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0]
        ]
    )
)
f1 = res1.wald_test(R1, V)
anticipation1 = False
att1 = '-'
