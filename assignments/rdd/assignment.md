# Regression Discontinuity Design
We will replicate the main results of the paper _Using Maimonides' Rule to Estimate the
Effect of Class Size on Scholastic Achievement_ (Angrist and Lavy, 1999). Our goal is to
estimate the causal effect of class size on test scores (see `./data/dict.md` for more
information on the model and dataset).

In class, we used a simple RDD model to estimate the effect of class size on average
math scores. Our model specification was:
```latex
$$
\text{math score}_i =
    \beta_0 + \beta_1 x_i + \beta_2 z_i + \beta_3 (z_i \times x_i)
$$
```
where `x_i` is the running variable (centered around zero) and `z_i` is the binary
treatment indicator (`z_i = 1` if there are 40 children or fewer in the _i_-th cohort).

We estimated `$\hat{\beta_2} = -6.0739$` and it was statistically significant. This
suggests that students just above the cutoff (who are therefore placed in smaller
classrooms) score, on average, 6.1 points higher in math than those at or below the
cutoff.

Since this is an RDD and we're estimating the effect at the cutoff, we interpret this as
the **Local Average Treatment Effect**. We concluded that reducing class size improves
math performance.

## Assignment
Your job is to estimate the LATE of class size on average verbal scores (`avgverb`).
Start off with a cubic model and trim it down as you see fit.

In RDD, we always allow the model to have heterogeneous slopes. In other words, we let
the treatment interact with the running variable even if one side is not significant.
However, if there is no evidence for a cubic term on either side of the discontinuity,
you should trim the model down to a second-degree polynomial (and so forth).

1. Create a new branch `assignment/rdd-<your ID here>`
2. Create a Jupyter Notebook where you fit your models (for your eyes only, don't push
this file)
3. Refactor your Notebook into a Python file (
    `./assignments/rdd/code/rdd-<your ID here>.py`
)
    - Make sure your final fitted model is named `res`
    - Make sure you use `HC3` standard errors
4. Add, commit and push your Python file
5. Make sure your submission passes the builds!

## Expected result
The estimated value of your final model's LATE should be `-5.3948` with a _p_-value of
`0.057`.
