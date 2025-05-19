# Difference in Differences (Card & Krueger, 1994)
This file is a dictionary of the data used in the paper *"Minimum Wages and Employment:
A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania"* (Card & Krueger,
1994). This article is the most famous application of a Difference-in-Differences
design.

## Context
In April 1992, the state of New Jersey (NJ) raised the minimum wage from $4.25/hr to
$5.05/hr, while the neighboring state of Pennsylvania (PA) maintained the minimum wage
at $4.25/hr. Classic economic theory suggests that increasing the minimum wage will
reduce employment, and this setting is a natural experiment to empirically test
that hypothesis.

The authors surveyed fast-food restaurants in PA and NJ before and after the policy.
Using a DiD design, they compared the average number of employees between both groups
and found that the salary increase did not reduce employment in NJ. This seemingly
counterintuitive result is often referred to as the Card-Krueger paradox.

## Columns
The file `../data/card-krueger.csv` contains the raw data of the original paper (with
some minor changes, such as replacing `'.'` with `null`, and converting the data from
wide to longitudinal fromat). The following lists describe the meaning of each column:

- `i`: unique store id
- `t`: time period (0=before policy, 1=after policy)
- `chain`: chain (1=bk; 2=kfc; 3=roy's; 4=wendy's)
- `co_owned`: 1 if company owned
- `state`: 1 if NJ; 0 if PA
- `nj_south`: 1 if in southern NJ
- `nj_central`: 1 if in central NJ
- `nj_north`: 1 if in northern NJ
- `nj_shore`: 1 if on NJ shore
- `pa_1`: 1 if in PA, northeast suburbs of Philadelphia
- `pa_2`: 1 if in PA, Easton, etc
- `type_1`: type of second interview (1=phone; 2=personal)
- `status_1`: status of second interview (0=refused, 1=answered, 2=renovation, 3=closed,
4=construction, 5=fire)
- `special`: 1 if special program for new workers
- `bonus`: 1 if cash bounty for new workers
- `pctaff`: % employees affected by new minimum wage
- `ncalls`: number of call-backs
- `empft`: number of full-time employees
- `emppt`: number of part-time employees
- `nmgrs`: number of managers/assistant managers
- `wagest`: starting wage ($/hr)
- `inctime`: months to usual first raise
- `firstinc`: usual amount of first raise ($/hr)
- `meals`: free/reduced price meals (0=none, 1=free, 2=reduced, 3=both)
- `open`: hour of opening
- `hrsopen`: number of hours open per day
- `psoda`: price of medium soda, including tax
- `pfry`: price of small fries, including tax
- `pentree`: price of entree, including tax
- `nregs`: number of cash registers in store
- `nregs11`: number of registers open at 11:00 am

The data is presented in long format, with two rows per store (one for each time
period).

## Transformation
The [original dataset](https://davidcard.berkeley.edu/data_sets/njmin.zip) is presented
in wide format (one row per store with two columns for each feature). The following code
was used to transform the original dataset to the file presented in this repo.

```python
# Load data
df = pd.read_csv('public.dat', sep='\s+', header=None)

# Rename columns
df.columns = [
    'i', 'chain', 'co_owned', 'state',
    'nj_south', 'nj_central', 'nj_north', 'pa_1', 'pa_2', 'nj_shore',
    'ncalls_0', 'empft_0', 'emppt_0', 'nmgrs_0', 'wagest_0', 'inctime_0',
    'firstinc_0', 'bonus', 'pctaff', 'meals_0', 'open_0', 'hrsopen_0',
    'psoda_0', 'pfry_0', 'pentree_0', 'nregs_0', 'nregs11_0',
    'type_1', 'status_1', 'date_1', 'ncalls_1', 'empft_1', 'emppt_1',
    'nmgrs_1', 'wagest_1', 'inctime_1', 'firstinc_1', 'special',
    'meals_1', 'open_1', 'hrsopen_1', 'psoda_1', 'pfry_1',
    'pentree_1', 'nregs_1', 'nregs11_1'
]

# Drop duplicated store
df = df[df['i'] != 407]

# Cast columns to correct data types
cols_str = list(df.dtypes[df.dtypes == object].index)
df[cols_str] = df[cols_str].replace('.', 'NaN').astype(float)

# Drop columns
df = df.drop(columns=['date_1'])

# Wide to long transformation
cols_varying = [
    'ncalls', 'empft', 'emppt', 'nmgrs', 'wagest', 'inctime', 'firstinc', 
    'meals', 'open', 'hrsopen', 'psoda', 'pfry', 'pentree', 'nregs', 'nregs11'
]
df = pd.wide_to_long(
    df=df,
    stubnames=cols_varying,
    i='i',
    j='t',
    sep='_',
    suffix='(0|1)'
).reset_index().sort_values(['i', 't'])

# Write data
df.to_csv('card-krueger.csv', index=False)
```
