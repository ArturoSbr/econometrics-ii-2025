# Importing libraries
import os
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# 1.Loading Data
PATH = os.path.join('..', 'data', 'callaway-santanna.csv')
df = pd.read_csv(PATH)

# 2. Rename Columns
df.rename(columns={'year': 't', 'countyreal': 'i', 'first.treat': 'treat_start'}, inplace=True)

# 3. Decalre new time column k
df['treat_start'] = df['treat_start'].replace(0, np.nan)
df['k'] = df['t'] - df['treat_start']

# 4. Set multi-index
df = df.set_index(["i", "t"])

# 4. Create dummies for event periods
dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)

# 5. Prepare panel dataframe
df = pd.concat([df, dummies], axis=1)

# 6. ATT (only treated countries)

# Mask_treated: ever-treated counties
mask_treated = ~df['treat_start'].isna()
# Using the Mask_treated, filter the dataframe
df_treated = df.loc[mask_treated]
# identify the columns needed (k_-1 is the reference Value)
event_cols = [c for c in df_treated.columns 
              if c.startswith('k_') and c not in ('k_-1.0', 'k_nan')]
#Fitting the model
mod0 = PanelOLS(dependent = df_treated['lemp'],
                exog = df_treated[event_cols],
                entity_effects = True,
                time_effects = True,
                drop_absorbed  = True)
res0 = mod0.fit(cov_type = 'clustered', cluster_entity = True)    

# 7.Check for anticipated effects

# Grab the list of estimated parameter names from res0
param_names = res0.params.index.tolist()
# Initialize a 3×P zero‐matrix, where P = number of estimated parameters
restriction = np.zeros((3, len(param_names)))
# Fill each row of R so that it picks off the correct β_k
for row, k in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    col_idx = param_names.index(k)
    restriction[row, col_idx] = 1
# Create the RHS vector of zeros (3×1)
values = np.zeros((3, 1))
# Run the Wald test
f0 = res0.wald_test(restriction, values)

# 8.Is there evidence of anticipation effects?
anticipation0=False

# 9.Is the effect of increasing the minimum wage positive or negative?
att0 = '+'

# 10.ATT (all units)

# Identify the columns needed (k_-1 is the reference Value)
event_cols = [c for c in df_treated.columns 
              if c.startswith('k_') and c not in ('k_-1.0')]
#Fitting the model
mod1 = PanelOLS(dependent = df['lemp'],
                exog = df[event_cols],
                entity_effects = True,
                time_effects = True,
                drop_absorbed  = True)
res1 = mod1.fit(cov_type = 'clustered', cluster_entity = True)    

# 11.Check for anticipated effects

# Grab the list of estimated parameter names from res0
param_names = res1.params.index.tolist()
# Initialize a 3×P zero‐matrix, where P = number of estimated parameters
restriction = np.zeros((3, len(param_names)))
# Fill each row of R so that it picks off the correct β_k
for row, k in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    col_idx = param_names.index(k)
    restriction[row, col_idx] = 1
# Create the RHS vector of zeros (3×1)
values = np.zeros((3, 1))
# Run the Wald test
f1 = res1.wald_test(restriction, values)


# 12.Is there evidence of anticipation effects?
anticipation1=False

# 13.Is the effect of increasing the minimum wage positive or negative?
att1 = '-'
