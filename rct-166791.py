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
    df[['ireland', 'scotland', 'wales', 'dark', 'dark_mobile']], 
    has_constant='add'
    )

spec = sm.OLS(
    endog=df['purchase'],
    exog=X1
)
model = spec.fit()
model.summary()

X2 = sm.add_constant(
    df[['ireland', 'scotland', 'wales', 'dark']], 
    has_constant='add'
    )
spec2 = sm.OLS(
    endog=df['purchase'],
    exog=X2
)
model2 = spec2.fit()
model2.summary()
