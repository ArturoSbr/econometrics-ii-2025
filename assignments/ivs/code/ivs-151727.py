# Importa librerias
import os
import pandas as pd
import numpy as np
from linearmodels import IV2SLS
import itertools
from statsmodels.regression.linear_model import OLS

# Global variables
df = None
res0 = None
res1 = None
bias = None
bias_sign = None

# Loads database
def load_data(file_name='raw.csv'):
    # Get the directory where THIS script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to 'ivs' directory, then into 'data'
    data_dir = os.path.join(script_dir, '..', 'data')
    data_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    return pd.read_csv(data_path)

# Creates dummies and interactions 
def create_dummies(df, start_year=1940, end_year=1950):
    # Filter data but keep all years in dummies
    df = df[df['yob'] >= start_year].copy()
    # Create dummies for full year range regardless of data existence
    yob_dummies = pd.get_dummies(df['yob'], prefix='yob').reindex(columns=[f'yob_{y}' for y in range(start_year, end_year + 1)], fill_value=0).astype(int)
    qob_dummies = pd.get_dummies(df['qob'], prefix='qob').astype(int)
    return pd.concat([df, yob_dummies, qob_dummies], axis=1)

def create_interactions(df, start_year=1940, end_year=1950):
    interactions = {}
    for year, quarter in itertools.product(range(start_year, end_year + 1), range(1, 5)):
        col_name = f'yob_{year}_qob_{quarter}'
        interactions[col_name] = (df[f'yob_{year}'] * df[f'qob_{quarter}']).astype(int)
    return pd.concat([df, pd.DataFrame(interactions)], axis=1)

# Runs table vi naive and iv (2sls)
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

# Checks bias 
def analyze_bias(naive_results, iv_results, threshold=0.01):
    naive_coef = naive_results.params['educ']
    iv_coef = iv_results.params['educ']
    bias = bool(abs(naive_coef - iv_coef) > threshold)
    bias_sign = '+' if naive_coef > iv_coef else '-' if bias else '0'
    return bias, bias_sign, naive_coef, iv_coef

# Code Execution
df = load_data()
df = df[df['cohort'] == '40-49']
df = create_dummies(df)
df = create_interactions(df)
df['const'] = 1

res0 = run_naive_model(df)
res1 = run_iv_model(df)
bias, bias_sign, naive_coef, iv_coef = analyze_bias(res0, res1)

print("\nNaive Model Results:")
print(res0.summary().tables[1])
print("\nIV Model Results:")
print(res1.summary.tables[1])
print(f"\nBias: {bias}, Direction: {bias_sign}")
