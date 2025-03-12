import os
import pandas as pd
import statsmodels.api as sm
PATH = os.path.join('..', 'data', 'raw.csv')
df = pd.read_csv(PATH)
df.columns = ['id', 'dark', 'views', 'time', 'purchase', 'mobile', 'location']
df.replace(
    to_replace={
        'dark': {'A': '0', 'B': '1'},
        'mobile': {'Mobile': '1', 'Desktop': '0'},
        'purchase': {'No': '0', 'Yes': '1'},
        'location': {'Northern Ireland': 'Ireland'}
    },
    inplace=True
)
df[['dark', 'mobile', 'purchase']] = df[['dark', 'mobile', 'purchase']].astype(int)

df['location'] = df['location'].str.lower()
df = pd.get_dummies(
    data=df,
    prefix='',
    prefix_sep='',
    columns=['location'],
    dtype=int
)
df['dark_mobile'] = df['dark']*df['mobile']

X1 = sm.add_constant(
    df[['ireland', 'scotland', 'wales', 'dark']]
    )

spec = sm.OLS(
    endog=df['purchase'],
    exog=df[['ireland', 'scotland', 'wales', 'dark']].assign(
        const = 1
    ),
    hasconst=True
)
model = spec.fit()
model.summary()
