import os
import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
df = pd.read_csv(os.path.join('..', 'data', 'callaway-santanna.csv'))
df = df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
})
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']
df = df.set_index(['i', 't'])
dummies = pd.get_dummies(
    df['k'],
    prefix='k',
    dummy_na=True
).astype(int)
df = df.join(dummies)
mask_treated = ~df['treat_start'].isna()
df_tr = df.loc[mask_treated]
event_cols = [c for c in df_tr.columns
              if c.startswith('k_') 
                 and c not in ('k_-1.0', 'k_nan')]
m0 = PanelOLS(
    dependent      = df_tr['lemp'],
    exog           = df_tr[event_cols],
    entity_effects = True,
    time_effects   = True,
    drop_absorbed  = True
)
res0 = m0.fit(
    cov_type       = 'clustered',
    #cluster_entity = True 
)
R = np.zeros((3, 6))
R[0, 0] = 1   
R[1, 1] = 1   
R[2, 2] = 1
values = np.zeros(3)
f0 = res0.wald_test(R, values)
anticipation0 = False
att0 = '-'
event_cols = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
m1 = PanelOLS(
    dependent      = df['lemp'],
    exog           = df[event_cols],
    entity_effects = True,
    time_effects   = True,
    drop_absorbed  = True    
)
res1 = m1.fit(
    cov_type       = 'clustered',
    cluster_entity = True
)
R = np.zeros((3, 7))
R[0, 0] = 1  
R[1, 1] = 1   
R[2, 2] = 1
values = np.zeros(3)
f1 = res1.wald_test(R, values)
anticipation1 = False
att1 = '-'