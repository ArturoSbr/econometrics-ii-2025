# Instrumental Variables
[This dataset](
    https://economics.mit.edu/sites/default/files/inline-files/NEW7080_1.rar
) was taken from [Angrist's Data Archive (MIT)](
    https://economics.mit.edu/people/faculty/josh-angrist/angrist-data-archive
) and was used in Angrist and Krueger (1991).

## Columns
- **age**: Age of the respondent at the time of the survey
- **ageq**: Age plus quarter of birth (in decimal form)
- **ageqsq**: Square of `ageq`
- **cohort**: Categorical label for decade of birth (`'20-29'`, `'30-39'`, `'40-49'`)
- **educ**: Years of completed education (endogenous variable)
- **enocent**: Dummy for employment in the **Northeast** non-central city
- **esocent**: Dummy for employment in the **South** central city
- **lwklywge**: Natural log of weekly wage (dependent variable)
- **married**: 1 if married, 0 otherwise
- **midatl**: Dummy for residence in the **Middle Atlantic** (NY, NJ, PA)
- **mt**: Dummy for residence in the **Mountain** region (MT, ID, WY, etc.)
- **neweng**: Dummy for residence in **New England** (MA, CT, etc.)
- **census**: Census year used for the observation
- **qob**: Quarter of birth (1 = Jan–Mar, 2 = Apr–Jun, etc.)
- **race**: 1 = white, 0 = non-white
- **smsa**: Urban indicator
- **soatl**: Dummy for residence in the **South Atlantic** (VA, NC, etc.)
- **wnocent**: Dummy for employment in the **West** non-central city
- **wsocent**: Dummy for employment in the **West** central city
- **yob**: Year of birth

## Steps followed to clean the data
The raw data has no column names and a few other details. The following steps were
taken to clean the original dataset.

```python
# Imports
import numpy as np
import pandas as pd

# Read data
df = pd.read_stata("NEW7080.dta")

# Rename columns
df.rename(
    columns={
        'v1': 'age',
        'v2': 'ageq',
        'v4': 'educ',
        'v5': 'enocent',
        'v6': 'esocent',
        'v9': 'lwklywge',
        'v10': 'married',
        'v11': 'midatl',
        'v12': 'mt',
        'v13': 'neweng',
        'v16': 'census',
        'v18': 'qob',
        'v19': 'race',
        'v20': 'smsa',
        'v21': 'soatl',
        'v24': 'wnocent',
        'v25': 'wsocent',
        'v27': 'yob'
    },
    inplace=True
)

# Drop unused columns
df.drop([col for col in df.columns if col.startswith('v')], axis=1, inplace=True)

# Create columns
df = df.assign(
    yob=np.where(
        df['yob'].le(1900),
        df['yob'] + 1900,
        df['yob']
    ),
    cohort=np.where(
        df['yob'].between(30, 39),
        '30-39',
        np.where(
            df['yob'].between(40, 49),
            '40-49',
            '20-29'
        )
    ),
    ageq=np.where(
        df['census'].eq(80),
        df['ageq'] - 1900,
        df['ageq']
    ),
    ageqsq=df['age'].pow(2)
)
```
