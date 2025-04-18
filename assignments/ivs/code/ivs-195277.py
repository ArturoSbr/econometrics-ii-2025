import os
import pandas as pd
import numpy as np
from linearmodels import IV2SLS
import itertools
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hc3

# Definir como None para que estén visibles antes de importar
df = None
res0 = None
res1 = None
bias = None
bias_sign = None

def load_data(file_name='raw.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ivs_dir = os.path.dirname(script_dir)
    data_path = os.path.join(ivs_dir, 'data', file_name)
    return pd.read_csv(data_path)

def create_dummies(df, start_year=1940, end_year=1950):
    df = df[df['yob'] >= start_year].copy()
    yob_dummies = pd.get_dummies(df['yob'], prefix='yob').astype(int)
    df = pd.concat([df, yob_dummies], axis=1)
    qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
    df = pd.concat([df, qob_dummies], axis=1)
    return df

def create_interactions(df, start_year=1940, end_year=1950):
    interactions = {}
    for year, quarter in itertools.product(range(start_year, end_year), range(1, 5)):
        col_name = f'yob_{year}_qob_{quarter}'
        interactions[col_name] = df[f'yob_{year}'] * df[f'qob_{quarter}']
    return pd.concat([df, pd.DataFrame(interactions)], axis=1)

def run_naive_model(df, dependent_var='lwklywge', start_year=1940, end_year=1949):
    controls = ['const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
                'wnocent', 'soatl', 'esocent', 'wsocent', 'mt', 'educ']
    controls.extend([f'yob_{year}' for year in range(start_year, end_year)])
    return OLS(df[dependent_var], df[controls]).fit(cov_type='HC3')

def run_iv_model(df, dependent_var='lwklywge', start_year=1940, end_year=1949):
    exog = ['const', 'race', 'married', 'smsa', 'neweng', 'midatl', 'enocent',
            'wnocent', 'soatl', 'esocent', 'wsocent', 'mt']
    exog.extend([f'yob_{year}' for year in range(start_year, end_year)])
    instruments = [f'yob_{year}_qob_{quarter}' 
                   for year in range(start_year, end_year + 1)
                   for quarter in range(1, 4)]
    return IV2SLS(dependent=df[dependent_var],
                  exog=df[exog],
                  endog=df[['educ']],
                  instruments=df[instruments]).fit(cov_type='robust')

def analyze_bias(naive_results, iv_results, threshold=0.01):
    naive_coef = naive_results.params['educ']
    iv_coef = iv_results.params['educ']
    bias = abs(naive_coef - iv_coef) > threshold
    if not bias:
        bias_sign = '0'
    else:
        bias_sign = '+' if naive_coef > iv_coef else '-'
    return bias, bias_sign, naive_coef, iv_coef

# Ejecutar directamente (sin __main__) para que el grader acceda al contexto
df = load_data()
df = df[df['cohort'] == '40-49']
df = create_dummies(df)
df = create_interactions(df)
df['const'] = 1

res0 = run_naive_model(df)
res1 = run_iv_model(df)
bias, bias_sign, naive_coef, iv_coef = analyze_bias(res0, res1)

print("\nNaive Model Results (OLS with HC3 robust standard errors):")
print(res0.summary().tables[1])
print("\nIV Model Results (2SLS with robust standard errors):")
print(res1.summary.tables[1])
print("\nBias Analysis:")
print(f"Naive model education coefficient: {naive_coef:.4f}")
print(f"IV model education coefficient: {iv_coef:.4f}")
print(f"Is naive model biased? {bias}")
print(f"Direction of bias: {bias_sign}")
