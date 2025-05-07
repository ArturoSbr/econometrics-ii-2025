# Difference-in-Differences Assignment

## Before you start
1. Switch to branch `develop` (`git checkout develop`)
2. Update the branch (`git pull`)
3. Create your own branch (`git checkout -b assignment/did-<your-id-here>`)
4. Save your solution in `./assignments/did/code/did-<your-id-here>.py`
5. Add and commit your python file
6. Push your python file to your branch (`git push`)

## Context
We will use an event study to analyze how raising the minimum wage affected 
teenage employment in the United States between 2003 and 2007 (Callaway & Sant'Anna,
2021). The dataset contains yearly observations of teenage employment levels across
multiple counties. For more info check `assignments/did/data/dict-callaway-santanna.md`.

## Questions

0. Import `os`, `numpy as np`, `pandas as pd` and `PanelOLS`

1. Load data

Load the data in an object named `df`. **Remember to use relative paths!**

2. Rename columns

Rename "year" to "t", "countyreal" to "i" and "first.treat" to "treat_start".

3. Declare new time column `k`
    - Set `treat_start` to `np.nan` for never-treated (control) counties
    - Define `k` as `t - treat_start`

4. Set multi-index

Set columns `i` and `t` as the index for `df` (in that order!).

5. Create dummies for event periods

One-hot encode column `k` into multiple columns with `pd.get_dummies` of type `int`:
- `'k_-4.0'`
- `'k_-3.0'`
- `'k_-2.0'`
- `'k_-1.0'`
- `'k_0.0'`
- `'k_1.0'`
- `'k_2.0'`
- `'k_3.0'`

6. ATT (only treated counties)

Estimate the model

$$
    Y_{it} = \alpha_i + \lambda_t +
    \sum_{k=k_{min}}^{k_{max}} \beta_k \mathbf{1}
    \{t - G_i = k\} + \varepsilon_{it}
$$

- Use only the treated population (hint: use a mask)
- Use `lemp` as the dependent variable
- Set $\beta_{-1}$ as the reference level
- Use `entity_effects` and `time_effects`
- Set `drop_absorbed=True`
- Your fitted model should be named `res0`
- Use `cov_type='clustered'` when fitting the model

7. Check for anticipated effects

Run a Wald test to test $H_0: \beta_{-4} = \beta_{-3} = \beta_{-2} = 0$ using
`f0 = res0.wald_test(restriction, values)`, where:
- `restriction` is a $3 \times 6$ matrix (there are six parameters because $\beta_3$ is
absorbed)
- `values` is a $3 \times 1$ vector

such that $\mathit{R} \beta = [0, 0, 0]^T$

8. Is there evidence of anticipation effects?

- Declare `anticipation0` as `True` or `False` according to your previous answer.

9. Is the effect of increasing the minimum wage positive or negative?

- Declare `att0` as `'+'` or `'-'` accordingly.

10. ATT (all units)

Estimate the model

$$
    Y_{it} = \alpha_i + \lambda_t +
    \sum_{k=k_{min}}^{k_{max}} \beta_k \mathbf{1}
    \{t - G_i = k\} + \varepsilon_{it}
$$

- Use `lemp` as the dependent variable
- Set $\beta_{-1}$ as the reference level
- Use `entity_effects` and `time_effects`
- Your fitted model should be named `res1`
- Use `cov_type='clustered'` when fitting the model

11. Check for anticipated effects

Run a Wald test to test $H_0: \beta_{-4} = \beta_{-3} = \beta_{-2} = 0$ using
`f1 = res1.wald_test(restriction, values)`, where:
- `restriction` is a $3 \times 7$ matrix
- `values` is a $3 \times 1$ vector

such that $\mathit{R} \beta = [0, 0, 0]^T$

12. Is there evidence of anticipation effects?

- Declare `anticipation1` as `True` or `False` according to your previous answer.

13. Is the effect of increasing the minimum wage positive or negative?

- Declare `att1` as `'+'` or `'-'` accordingly.
