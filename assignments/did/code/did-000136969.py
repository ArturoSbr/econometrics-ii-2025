import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load data
df = pd.read_csv('../data/callaway-santanna.csv')


# 2. Rename columns
df = df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
})

# 3. Declare new time column k
#    – set treat_start = NaN for never‐treated
df['treat_start'] = df['treat_start'].replace(0, np.nan)
#    – define k = t − treat_start
df['k'] = df['t'] - df['treat_start']

# 4. Set multi‐index (i, t)
df = df.set_index(['i', 't'])

# 5. Create dummies for each event‐time (including NaN)
dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)
df = pd.concat([df, dummies], axis=1)

# Identify all the event‐dummy columns
event_cols = sorted(c for c in df.columns if c.startswith('k_'))