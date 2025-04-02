import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read data
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)

# View distributions
cols = ['cohsize', 'avgmath', 'avgverb', 'tipuach']
for col in cols:
    plt.hist(df[col], bins=20)
    plt.title(col.title())
    plt.show()

# Plot data
plt.axvline(40, color='gray', ls='--', lw=2, zorder=1)
plt.scatter(
    x=df.loc[df['z'].eq(1), 'cohsize'],
    y=df.loc[df['z'].eq(1), 'avgverb'],
    color='C1',
    alpha=0.6,
    label='Large class size (Z=1)',
    zorder=2
)
plt.scatter(
    x=df.loc[df['z'].eq(0), 'cohsize'],
    y=df.loc[df['z'].eq(0), 'avgverb'],
    color='C0',
    alpha=0.6,
    label='Small class size (Z=0)',
    zorder=2
)

# Add naive means for each group
plt.axhline(
    df.loc[df['z'].eq(1), 'avgverb'].mean(), xmax=0.47, color='C1', zorder=3
)
plt.axhline(
    df.loc[df['z'].eq(0), 'avgverb'].mean(), xmin=0.47, xmax=1, color='C0', zorder=3
)

# Aesthetics
plt.xticks(np.arange(30, 51, 2))
plt.ylim(0, 100)
plt.title('Cohort Size vs. Average Verbal Score')
plt.xlabel('Cohort Size')
plt.ylabel('Average Verbal Score')
plt.legend(loc='lower right')

# Show
plt.show()

# Fit polynomial model
# Feature engineering
CUTOFF = 40
df['x'] = df['cohsize'] - CUTOFF

# Declare and fit model
spec = sm.OLS.from_formula(formula='avgverb ~ 1 + x + + z + z * x', data=df)
res = spec.fit(cov_type='HC3')

# View results
print(res.summary())

# Plot same graph as before but instead of using naive averages, plot the RDD model from before.
# Data to plot
to_plot = df.assign(
    pred_m1=res.predict(df)
).sort_values('x')

# Plot
plt.axvline(0, color='gray', ls='--', lw=2, zorder=1)
plt.scatter(
    x=df.loc[df['z'].eq(1), 'x'],
    y=df.loc[df['z'].eq(1), 'avgverb'],
    color='C1',
    alpha=0.6,
    label='Large class size (Z=1)',
    zorder=2
)
plt.scatter(
    x=df.loc[df['z'].eq(0), 'x'],
    y=df.loc[df['z'].eq(0), 'avgverb'],
    color='C0',
    alpha=0.6,
    label='Small class size (Z=0)',
    zorder=2
)
plt.plot(
    to_plot.loc[to_plot['z'].eq(0), 'x'],
    to_plot.loc[to_plot['z'].eq(0), 'pred_m1'],
    lw=3, color='C0'
)
plt.plot(
    to_plot.loc[to_plot['z'].eq(1), 'x'],
    to_plot.loc[to_plot['z'].eq(1), 'pred_m1'],
    lw=3, color='C1'
)

# Aesthetics
plt.ylim(0, 100)
plt.title('Cohort Size vs. Average Verbal Score')
plt.xlabel('Cohort Size - Cutoff')
plt.ylabel('Average Verbal Score')
plt.legend(loc='lower right')

# Show
plt.show()
plt.show()