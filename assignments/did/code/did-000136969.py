import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load data
data_path = os.path.join('assignments', 'did', 'data', 'callaway_santanna.csv')
df = pd.read_csv(data_path)

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

# ------------------------------------------------------------------------------
# 6. ATT (only treated counties)
# ------------------------------------------------------------------------------

# use only treated counties
df_t = df[df['treat_start'].notna()]

# drop the reference period β₋₁
exog_t = df_t[event_cols].drop(columns=['k_-1.0'])
y_t = df_t['lemp']

# fit the event‐study with FE and clustered SEs
res0 = PanelOLS(
    y_t,
    exog_t,
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

# 7. Wald test H₀: β₋₄ = β₋₃ = β₋₂ = 0
# build the 3×K restriction matrix
K0 = len(res0.params)
R0 = np.zeros((3, K0))
idx0 = {name: i for i, name in enumerate(res0.params.index)}

for j, kname in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    R0[j, idx0[kname]] = 1

vals0 = np.zeros(3)
f0 = res0.wald_test(R0, vals0)

# 8. Evidence of anticipation?
anticipation0 = bool(f0.pval < 0.05)

# 9. Sign of the treatment effect at t=0
att0 = '+' if res0.params['k_0.0'] > 0 else '-'

# ------------------------------------------------------------------------------
# 10. ATT (all units)
# ------------------------------------------------------------------------------

exog_all = df[event_cols].drop(columns=['k_-1.0'])
y_all = df['lemp']

res1 = PanelOLS(
    y_all,
    exog_all,
    entity_effects=True,
    time_effects=True
).fit(cov_type='clustered', cluster_entity=True)

# 11. Wald test H₀: β₋₄ = β₋₃ = β₋₂ = 0 (all units)
K1 = len(res1.params)
R1 = np.zeros((3, K1))
idx1 = {name: i for i, name in enumerate(res1.params.index)}

for j, kname in enumerate(['k_-4.0', 'k_-3.0', 'k_-2.0']):
    R1[j, idx1[kname]] = 1

vals1 = np.zeros(3)
f1 = res1.wald_test(R1, vals1)

# 12. Evidence of anticipation?
anticipation1 = bool(f1.pval < 0.05)

# 13. Sign of the treatment effect at t=0
att1 = '+' if res1.params['k_0.0'] > 0 else '-'

# -- end of script ------------------------------------------------------------

# You can print a quick summary:
print(res0.summary)
print(f"Anticipation (treated only)? {anticipation0}")
print(f"Sign of ATT₀: {att0}\n")

print(res1.summary)
print(f"Anticipation (all units)? {anticipation1}")
print(f"Sign of ATT₁: {att1}")
