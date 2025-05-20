# Q0. Imports
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# Q1. Load data
df = pd.read_csv('../data/callaway-santanna.csv')
print(df.head())

# Q2. Rename columns: year→t, countyreal→i, first.treat→treat_start
df = df.rename(columns={
    'year': 't',
    'countyreal': 'i',
    'first.treat': 'treat_start'
})
print(df.head())

# Q3. Declare event‐time k
df['treat_start'] = df['treat_start'].replace(0, np.nan)      # never‐treated → NaN
df['k'] = df['t'] - df['treat_start']                         # k = t − treat_start
print(df.head())

# Q4. Set multi‐index (i, t)
df = df.set_index(['i', 't'])
print(df.head())

# Q5. Create dummies for each event‐time (including NaN)
kd = pd.get_dummies(df['k'], prefix='k', dummy_na=True).astype(int)
df = df.join(kd)

# Q6. ATT (only treated & balanced panel)
mt = df['treat_start'].notna()                                # mask: treated
mf = ~df['lemp'].isna().groupby(level='i').any()               # mask: balanced (no missing lemp)
valid_ids = mf[mf].index
mask0 = mt & df.index.get_level_values('i').isin(valid_ids)
df0 = df[mask0].copy()

# select event‐time dummies excluding reference k_-1.0 and zero‐var columns
k_cols0 = [
    c for c in df0.columns
    if c.startswith('k_') and c != 'k_-1.0' and df0[c].nunique() > 1
]
ex0 = df0[k_cols0]

spec0 = PanelOLS(
    df0['lemp'],
    ex0,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res0 = spec0.fit(cov_type='clustered')
print(res0.summary)

# Q7. Wald test H0: β₋₄ = β₋₃ = β₋₂ = 0
R0 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])
v0 = np.zeros(3)
f0 = res0.wald_test(R0, v0)
print('Wald f0:', f0)

# Q8. Anticipation effects?
anticipation0 = False

# Q9. Sign of ATT?
att0 = '-'

# Q10. ATT (all units)
k_cols1 = [c for c in df.columns if c.startswith('k_') and c != 'k_-1.0']
ex1 = df[k_cols1]

spec1 = PanelOLS(
    df['lemp'],
    ex1,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
)
res1 = spec1.fit(cov_type='clustered')
print(res1.summary)

# Q11. Wald test H0: β₋₄ = β₋₃ = β₋₂ = 0 (all units)
R1 = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0]
])
v1 = np.zeros(3)
f1 = res1.wald_test(R1, v1)
print('Wald f1:', f1)

# Q12. Anticipation effects?
anticipation1 = False

# Q13. Sign of ATT?
att1 = '-'
