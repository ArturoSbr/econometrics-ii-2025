import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load data
# Como estamos en assignments/did/code, subimos un nivel (..) y vamos a data/
data_path = os.path.join('..', 'data', 'callaway-santanna.csv')
df = pd.read_csv(data_path)

# 2. Rename columns
df = df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
})

# 3. Declare new time column k
df['treat_start'] = df['treat_start'].replace(0, np.nan)
df['k'] = df['t'] - df['treat_start']

# 4. Set multi‐index (i, t)
df = df.set_index(['i', 't'])

# 5. Create dummies for event periods (incluye nan)
dummies = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)
df = pd.concat([df, dummies], axis=1)
event_cols = sorted(c for c in df.columns if c.startswith('k_'))