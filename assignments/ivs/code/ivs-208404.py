import pandas as pd
import statsmodels.api as sm
import itertools

df = pd.read_csv('../data/raw.csv')

df = df[df['yob'] >= 1940]

yob_dummies = pd.get_dummies(df['yob'], prefix='yob')
yob_dummies = yob_dummies.loc[:, [f'yob_{y}' for y in range(1940, 1950)]]

qob_dummies = pd.get_dummies(df['qob'], prefix='qob')
qob_dummies = qob_dummies.loc[:, [f'qob_{q}' for q in range(1, 5)]]
yob_dummies = yob_dummies.astype(int)
qob_dummies = qob_dummies.astype(int)

df = pd.concat([df, yob_dummies, qob_dummies], axis=1)
yob_cols = [f'yob_{y}' for y in range(1940, 1950)]
qob_cols = [f'qob_{q}' for q in range(1, 5)]

interaction_cols = {
    f'{y}_{q}': df[y] * df[q]
    for y, q in itertools.product(yob_cols, qob_cols)
}
df = df.assign(**interaction_cols)
formula = (
    "lwklywge ~ educ + race + married + smsa + "
    "neweng + midatl + enocent + wnocent + soatl + "
    "esocent + wsocent + mt + C(yob, Treatment(1949))"
)
model = sm.OLS.from_formula(formula, data=df)
res0 = model.fit(cov_type='HC3')
controls = [
    'race', 'married', 'smsa',
    'neweng', 'midatl', 'enocent', 'wnocent',
    'soatl', 'esocent', 'wsocent', 'mt'
]
yob_controls = [f'yob_{y}' for y in range(1940, 1949)]  # reference = 1949
controls += yob_controls
instr_vars = [
    f'yob_{y}_qob_{q}'
    for y in range(1940, 1950)
    for q in range(1, 4)  # omit qob_4
    if f'yob_{y}_qob_{q}' in df.columns
]

X1 = df[instr_vars + controls]
X1 = sm.add_constant(X1, has_constant='add')
y1 = df['educ']
first_stage = sm.OLS(y1, X1).fit()
df['educ_hat'] = first_stage.fittedvalues
X2 = df[controls + ['educ_hat']]
X2 = sm.add_constant(X2, has_constant='add')
y2 = df['lwklywge']
second_stage = sm.OLS(y2, X2).fit(cov_type='HC3')
res1 = second_stage
bias = True
bias_sign = '+'