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

## Instructions
1. Load the data in an object named `df`.
2. Filter out respondants born before 1940 (keep 1940 onward).
3. Create a dummy for year of birth and quarter of birth (use `pd.get_dummies` to make
your life easier). The resulting variables must be named `yob_1940`, `qob_1`, etc.
    - Make sure the values are integers and not booleans!
4. Create an interaction between all years and all quarters. For example, create
`df['yob_1940_qob_1'] = df['yob_1940'] * df['qob_1']`.
    - Hint: Ask an AI copilot to help you create these columns with `itertools` and
    `pd.assign`.
