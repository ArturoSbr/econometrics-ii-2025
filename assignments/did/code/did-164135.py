import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
PATH = os.path.join('..','data', 'callaway-santanna.csv')
df = pd.read_csv(PATH)
df = df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
})
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']
df = df.set_index(['i', 't'])
df_dummies = pd.get_dummies(df['k'], prefix='k', dtype=int, dummy_na=True)
df = pd.concat([df, df_dummies], axis=1)
treated_mask = df['treat_start'].notna()  
df_treated = df[treated_mask] 
y = df_treated['lemp']
X = df_treated.drop(columns=['lemp','k_-1.0','lpop','treat_start','treat','k','k_nan'])
model = PanelOLS(y, X, entity_effects=True, time_effects=True,drop_absorbed=True)
res0 = model.fit(cov_type='clustered')
restriction = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0],  
])
values = np.array([0, 0, 0])
f0 = res0.wald_test(restriction, value=values)
anticipation0=False
y_2 = df['lemp']
X_2 = df.drop(columns=['lemp','k_-1.0','lpop','treat_start','treat','k','k_nan'])
model_1 = PanelOLS(y_2, X_2, entity_effects=True, time_effects=True,drop_absorbed=True)
res1 = model_1.fit(cov_type='clustered')
restriction_1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0],  
])
values_1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction_1, value=values_1)
anticipation1=False
att1='-'