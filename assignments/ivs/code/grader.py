# Imports
import importlib.util
import sys
import traceback
from pathlib import Path
import os

# Find student's script
def find_student_script():
    for filename in os.listdir():
        if filename.startswith("ivs-") and filename.endswith(".py"):
            return filename
    raise FileNotFoundError("No student script matching 'ivs-<id>.py' found.")
    sys.exit(1)

# Import student's objects
def import_student_solution(path):
    spec = importlib.util.spec_from_file_location("student_solution", path)
    student = importlib.util.module_from_spec(spec)
    sys.modules["student_solution"] = student
    spec.loader.exec_module(student)
    return student

# Function that runs iterable
def run_test(step_num, test_code, context):
    print(f"Grading Q{step_num}")
    try:
        exec(test_code, context)
        print(f"Step {step_num} passed.")
        return 1
    except AssertionError as e:
        print(f"Step {step_num} failed: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Step {step_num} errored: {e}")
        traceback.print_exc()
    return 0

def main():
    try:
        student_path = find_student_script()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Found student script: {student_path}")
    student = import_student_solution(student_path)
    context = student.__dict__
    score = 0
    total = 0

    test_cells = [
        # Q1 — Check if df exists
        (1, "df"),

        # Q2 — Filter
        (2, """assert 'cohort' in df.columns, 'Missing `cohort` column'
assert df['cohort'].nunique() == 1, 'Cohort should contain only one value'
assert df['cohort'].unique().item() == '40-49', 'Cohort value should be "40-49"'
"""),

        # Q3 — Dummies for year and quarter of birth
        (3, """TEST_3_COLS = [
    'yob_1940', 'yob_1941', 'yob_1942', 'yob_1943', 'yob_1944',
    'yob_1945', 'yob_1946', 'yob_1947', 'yob_1948', 'yob_1949',
    'qob_1', 'qob_2', 'qob_3', 'qob_4'
]
try:
    df[TEST_3_COLS]
except:
    raise ValueError(f'One or more columns not found. Expected {TEST_3_COLS}')
assert df[TEST_3_COLS].dtypes.astype(str).unique().item() == 'int64', (
    f'Columns {TEST_3_COLS} must be integer type (int64).'
)
"""),

        # Q4 — Interactions
        (4, """import itertools
yob = [f'yob_{str(i)}' for i in range(1940, 1950)]
qob = [f'qob_{str(i)}' for i in range(1, 5)]
prods = list(itertools.product(yob, qob))
TEST_4_COLS = [f'{y}_{q}' for y, q in prods]
try:
    df[TEST_4_COLS]
except:
    raise ValueError(f'One or more columns not found. Expected {TEST_4_COLS}')
assert df[TEST_4_COLS].dtypes.astype(str).unique().item() == 'int64', (
    f'Columns {TEST_4_COLS} must be integer type (int64)'
)
"""),

        # Q5 — OLS t-stat
        (5, """TEST_5_T = 149.8684189656074
assert np.abs(res0.tvalues['educ'].item() - TEST_5_T) < 1e-6, (
    f't-stat for `educ` does not match. Expected {TEST_5_T}.'
)
"""),

        # Q6 — IV t-stat
        (6, """TEST_6_T = 2.698848959270176
assert np.abs(res1.tstats['educ'].item() - TEST_6_T) < 1e-6, (
    f't-stat for `educ` does not match. Expected {TEST_6_T}.'
)
"""),

        # Q7 — Bias must be True
        (7, """assert isinstance(bias, bool), 'bias must be True or False'
assert bias, 'Think about why it makes sense to run an IV model!'
"""),

        # Q8 — Bias sign must be +
        (8, """assert bias_sign in ['0', '-', '+'], (
    'bias_sign must be one of "0", "+" or "-"'
)
assert bias_sign == '+', (
    'Think about whether the naive model over or underestimates the true effect of ',
    'education on wages!'
)
"""),
    ]

    for step_num, test_code in test_cells:
        score += run_test(step_num, test_code, context)
        total += 1

    print("\nFinal Score: {}/{} steps passed.".format(score, total))

if __name__ == "__main__":
    main()
