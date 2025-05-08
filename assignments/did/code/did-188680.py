#Imports 
import os 
import numpy as np 
from linearmodels.panel import PanelOLS
import pandas as pd 

#Load Data 
PATH = os.path.join('..', 'data', "callaway-santanna.csv")
df = pd.read_csv(PATH)
df.head()

#Rename columns 
df.rename(columns={
    "year": "t",
    "countyreal": "i",
    "first.treat": "treat_start"
}, inplace=True)

#Set never treated units to nan and declare new time column k 
df.loc[df['treat_start'] == 0, 'treat_start'] = np.nan
df['k'] = df['t'] - df['treat_start']

#Set Multi Index 
df.set_index(['i', 't'], inplace=True)

#Create Dummies 
k_dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True, dtype=int)
df = df.join(k_dummies)

#ATT (Only treated units)
mask_treated = df['treat_start'].notna()
mask_full_panel = ~df['lemp'].isna().groupby(level='i').any()
valid_units = mask_full_panel[mask_full_panel].index
mask = mask_treated & df.index.get_level_values('i').isin(valid_units)
df_att = df[mask].copy()

all_k_dummies = [col for col in df_att.columns if col.startswith('k_')]
k_dummies = [k for k in all_k_dummies if k != 'k_-1.0']
k_dummies_final = [k for k in k_dummies if df_att[k].nunique() > 1]
exog0 = df_att[k_dummies_final]

spec0 = PanelOLS(
    df_att['lemp'],
    exog0,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)

res0=spec0.fit(cov_type='clustered')

#Wald test res0
restriction0 = np.array([
    [1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0]   
])
values0 = np.array([0, 0, 0])  
f0=res0.wald_test(restriction0, values0)

#Anticipation Effects?
anticipation0= False
#Effect of increasing the minimum wage?
att0= '-'

#ATT (All units)

exog_vars1 = [col for col in df.columns if col.startswith('k_') and col != 'k_-1.0']
exog1 = df[exog_vars1]

spec1 = PanelOLS(
    df['lemp'],           
    exog1,
    entity_effects=True,  
    time_effects=True,    
    drop_absorbed=True    
)

res1=res1 = spec1.fit(cov_type='clustered')

#Wald test res1
restriction1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 0]   
])
values1 = np.array([0, 0, 0])
f1 = res1.wald_test(restriction1, values1)

#Anticipation Effects?
anticipation1=False

#Effect of increasing the minimum wage?
att1='-'




