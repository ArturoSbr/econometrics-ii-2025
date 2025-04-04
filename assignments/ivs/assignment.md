# Instrumental Variables Assignment
Your job is to replicate the main results presented in [Angrist and Krueger (1991)](
    https://www.jstor.org/stable/2937954
). In this paper, the authors study the effect of education on earnings. As seen in
class, a person's education level is affected by unobservable variables, such as
socioeconomic background, social skills, networking ability, etc. The authors therefore
instrument education using people's quarter of birth.

One's quarter of birth is considered to be as good as random and is used as an
instrument for education because in the US, children are required to turn six years old
by January 1st of the current school year and are required to stay in school until they
turn 16.

As a consequence, children who do not make the cutoff (i.e., children born in the first
quarter) must wait until next year to enroll in first grade. These children are
therefore older than the rest of their classmates and will turn 16 before everyone else.
In other words, they are able to legally drop out before other students.

## Before you start
1. Clone the repo if you haven't done so already
2. Switch to branch `develop`
3. Pull the latest version of `develop` (`git pull origin develop`)
4. Familiarize yourself with the dataset (read `assignments/ivs/data/dict.md`)
5. After updating your repo, switch to a new branch `assignments/ivs-<your id here>`
5. Submit a refactored python script (`assignments/ivs/code/ivs-<your id here>.py`) with
`git add`, `git commit` and `git push`
    - For example: `assignments/ivs/code/ivs-130524.py`
6. Remember **NOT TO USE** `git add *` or `git add .` (don't be that person)
    - Don't submit your notebook or anything that's not the refactored Python script

## Instructions
1. Load the data in an object named `df`
    - Your code should be in this directory: `./assignments/ivs/code/`, so use relative
    paths to get back to `/data`.
2. Filter out respondants born before 1940 (keep 1940 onward)
3. Create a dummy for year of birth and quarter of birth (use `pd.get_dummies` to make
your life easier). The resulting variables must be named `yob_1940`, ..., `yob_1949`,
`qob_1`, ..., `qob_4`.
    - Make sure the values are integers and not booleans!
4. Create an interaction term for every year and every quarter using the dummies you
created in the previous step. For example,
`df['yob_1940_qob_1'] = df['yob_1940'] * df['qob_1']`.
    - Hint: Ask an AI copilot to help you create these columns with `itertools` and
    `pd.assign`
    - The new columns must follow the naming convention `yob_{year}_qob_{quarter}`. For
    example, `yob_1944_qob_3`.
5. Run a naive model; one that doesn't account for `educ`'s endogeneity (remember to
add a constant to `df`):
    - Store the results in `res0`
    - Use robust standard errors (HC3)
    - Dependent variable: lwklywge
    - Controls: const, race, married, smsa, neweng, midatl, enocent, wnocent,
    soatl, esocent, wsocent, mt, educ and your year of birth dummies (set 1949 as the
    reference)
6. Run a two-stage least squares IV model to account for `educ`'s endogeneity using
`IV2SLS` from `linearmodels`:
    - Store the reults in `res1`
    - Dependent variable: lwklywge
    - Controls: const, race, married, smsa, neweng, midatl, enocent, wnocent,
    soatl, esocent, wsocent, mt and your year of birth dummies (set 1949 as the
    reference)
    - Endogenous variable: educ
    - Instruments: All the interactions between year of birth and quarter of birth (set
    each year's reference to the last quarter, for example, omit `yob_1940_qob_4`).
7. Does this evidence suggest that the naive model is biased?
    - Declare `bias = True` if you think the naive model is biased
    - Declare `bias = False` otherwise
8. Do you think the bias is negative, positive or zero (unbiased)?
    - Declare `bias_sign = '0'` if you think there's no bias
    - Declare `bias_sign = '+'` if you think it's positive
    - Declare `bias_sign = '-'` if you think it's negative
