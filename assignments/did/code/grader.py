import os
import sys
import importlib.util
import pandas as pd
import numpy as np
import traceback


# Import module
def import_student_module():
    student_file = None

    # Look for file matching did-*.py
    for file in os.listdir():
        if file.startswith('did-') and file.endswith('.py'):
            student_file = file
            break

    if student_file is None:
        raise FileNotFoundError("No files match 'did-<id>.py' in `code` directory.")

    filepath = os.path.abspath(student_file)
    spec = importlib.util.spec_from_file_location("student", filepath)
    student = importlib.util.module_from_spec(spec)
    sys.modules["student"] = student
    spec.loader.exec_module(student)

    return student


# Function that runs assertions
def try_assert(test_func, description, points=1):
    try:
        test_func()
        return points, f"PASS: {description}"
    except AssertionError as e:
        return 0, f"FAIL: {description} — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: {description} — {str(e)}"
    

# Q1 - df
def check_q1(student):
    df = getattr(student, 'df', None)
    description = "Q1: `df` should be a pandas DataFrame loaded from the data file"

    return try_assert(
        lambda: isinstance(df, pd.DataFrame),
        description
    )


# Q2 - columns
def check_q2(student):
    df = getattr(student, 'df', None)
    t2_expected = ['t', 'i', 'lpop', 'lemp', 'treat_start', 'treat']
    description = "Q2: DataFrame should have correctly renamed columns"

    return try_assert(
        lambda: list(df.columns) == t2_expected,
        f"{description}. Expected {t2_expected}, got {list(df.columns)}"
    )


# Q3 - column k
def check_q3(student):
    df = getattr(student, 'df', None)
    t3_expected_nan = 1545
    t3_expected_num = 955

    try:
        assert df is not None, "'df' not found in student submission"
        assert 'k_nan' in df.columns, (
            "'k_nan' column not found. Set `dummy_na=True` in pd.get_dummies!"
        )

        nan_count = df['k_nan'].sum()
        num_count = df.shape[0] - nan_count  # total minus nan

        assert nan_count == t3_expected_nan, (
            f"There should be {t3_expected_nan} NaNs in 'k'. Found {nan_count}."
        )
        assert num_count == t3_expected_num, (
            f"There should be {t3_expected_num} non-NaN 'k' values. Found {num_count}."
        )

        return 1, "PASS: Q3 — Correct handling of NaNs in 'k' (validated via 'k_nan')"
    except AssertionError as e:
        return 0, f"FAIL: Q3 — {str(e)}"
    except Exception as e:
        import traceback
        return 0, f"ERROR: Q3 — Unexpected error:\n{traceback.format_exc()}"


# Q4 - multi index
def check_q4(student):
    df = getattr(student, 'df', None)
    t4_expected = ['i', 't']

    try:
        assert list(df.index.names) == t4_expected, (
            f"Index names should be {t4_expected}, "
            f"but current index names are {list(df.index.names)}."
        )
        return 1, "PASS: Q4 — DataFrame has correct multi-index ['i', 't']"
    except AssertionError as e:
        return 0, f"FAIL: Q4 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q4 — Unexpected error: {str(e)}"


# Q5 - dummies
def check_q5(student):
    df = getattr(student, 'df', None)
    t5_expected = [
        'k_-4.0', 'k_-3.0', 'k_-2.0', 'k_-1.0',
        'k_0.0', 'k_1.0', 'k_2.0', 'k_3.0', 'k_nan'
    ]

    try:
        assert df is not None, "'df' not found in student submission"
        for col in t5_expected:
            assert col in df.columns, (
                f"Column '{col}' not found in DataFrame columns: {list(df.columns)}"
            )
        return 1, "PASS: Q5 — All expected dummy columns for 'k' are present"
    except AssertionError as e:
        return 0, f"FAIL: Q5 — {str(e)}"
    except Exception as e:
        import traceback
        return 0, f"ERROR: Q5 — Unexpected error:\n{traceback.format_exc()}"



# Q6 - m0 params
def check_q6(student):
    res0 = getattr(student, 'res0', None)
    t6_expected = [
        -0.04557637,
        -0.01446286,
        -0.00056342,
        0.00841206,
        0.00176215,
        -0.06986437
    ]

    try:
        import numpy as np
        assert res0 is not None, "Object 'res0' not found"
        assert np.allclose(res0.params.values, t6_expected), (
            "One or more estimates in 'res0' do not match the expected values.\n"
            f"Expected: {t6_expected}\n"
            f"Got: {res0.params.values.tolist()}"
        )
        return 1, "PASS: Q6 — Model on treated units (`res0`) matches coefficients"
    except AssertionError as e:
        return 0, f"FAIL: Q6 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q6 — Unexpected error: {str(e)}"


# Q7 - f0
def check_q7(student):
    f0 = getattr(student, 'f0', None)
    t7_expected = 0.45736874043815534

    try:
        assert f0 is not None, "Object 'f0' not found"
        pval = f0.pval.item()
        assert abs(t7_expected - pval) < 1e-6, (
            f"Your p-value ({pval}) differs from the expected value ({t7_expected})."
        )
        return 1, "PASS: Q7 — Correct p-value"
    except AssertionError as e:
        return 0, f"FAIL: Q7 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q7 — Unexpected error: {str(e)}"


# Q8 - anticipation0
def check_q8(student):
    anticipation0 = getattr(student, 'anticipation0', None)
    t7_expected = 0.45736874043815534

    try:
        assert anticipation0 is not None, "Variable 'anticipation0' not found"
        assert not anticipation0, (
            f"The correct p-value ({t7_expected}) suggests there are no anticipation "
            f"effects, but you set anticipation0 = {anticipation0}."
        )
        return 1, "PASS: Q8 — Correct interpretation of anticipation effects"
    except AssertionError as e:
        return 0, f"FAIL: Q8 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q8 — Unexpected error: {str(e)}"


# Q9 - ATT0
def check_q9(student):
    att0 = getattr(student, 'att0', None)
    t9_expected = '-'

    try:
        assert att0 is not None, "Variable 'att0' not found"
        assert att0 == t9_expected, (
            "Incorrect sign for ATT. You should analyze the coefficients in "
            "`res0.summary`."
        )
        return 1, "PASS: Q9 — Correct interpretation of treatment effect (att0 = '-')"
    except AssertionError as e:
        return 0, f"FAIL: Q9 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q9 — Unexpected error: {str(e)}"


# Q10 - m1
def check_q10(student):
    res1 = getattr(student, 'res1', None)
    t10_expected = [
        0.0035493269194322922,
        0.024623501986490125,
        0.023354814886402744,
        -0.01814392696658068,
        -0.043472372628553246,
        -0.13179485775431962,
        -0.0922467941818849
    ]

    try:
        import numpy as np
        assert res1 is not None, "Object 'res1' not found"
        assert np.allclose(res1.params.values, t10_expected), (
            "One or more estimates in 'res1' do not match the expected values.\n"
            f"Expected: {t10_expected}\n"
            f"Got: {res1.params.values.tolist()}"
        )
        return 1, "PASS: Q10 — Model on all units (`res1`) matches expected coefficients"
    except AssertionError as e:
        return 0, f"FAIL: Q10 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q10 — Unexpected error: {str(e)}"
    

# Q11 - f1
def check_q11(student):
    f1 = getattr(student, 'f1', None)
    t11_expected = 0.22394535311127095

    try:
        assert f1 is not None, "Object 'f1' not found"
        pval = f1.pval.item()
        assert abs(t11_expected - pval) < 1e-6, (
            f"Your p-value ({pval}) differs from the expected value ({t11_expected})."
        )
        return 1, "PASS: Q11 — Correct p-value"
    except AssertionError as e:
        return 0, f"FAIL: Q11 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q11 — Unexpected error: {str(e)}"


# Q12 - anticipation1
def check_q12(student):
    anticipation1 = getattr(student, 'anticipation1', None)
    t11_expected = 0.22394535311127095

    try:
        assert anticipation1 is not None, "Variable 'anticipation1' not found"
        assert not anticipation1, (
            f"The correct p-value ({t11_expected}) suggests there are no anticipation effects, "
            f"but you set anticipation1 = {anticipation1}."
        )
        return 1, "PASS: Q12 — Correct interpretation of anticipation effects (anticipation1 = False)"
    except AssertionError as e:
        return 0, f"FAIL: Q12 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q12 — Unexpected error: {str(e)}"


# Q13 - ATT1
def check_q13(student):
    att1 = getattr(student, 'att1', None)
    t13_expected = '-'

    try:
        assert att1 is not None, "Variable 'att1' not found"
        assert att1 == t13_expected, (
            "Incorrect sign for ATT. You should analyze the coefficients in `res1.summary` "
            "and determine whether the treatment effect is negative or positive."
        )
        return 1, "PASS: Q13 — Correct sign interpretation of treatment effect (att1 = '-')"
    except AssertionError as e:
        return 0, f"FAIL: Q13 — {str(e)}"
    except Exception as e:
        return 0, f"ERROR: Q13 — Unexpected error: {str(e)}"


# Final grade
def run_grader(student):
    checks = [
        check_q1, check_q2, check_q3, check_q4, check_q5,
        check_q6, check_q7, check_q8, check_q9, check_q10,
        check_q11, check_q12, check_q13
    ]

    total = 0
    for i, check in enumerate(checks, start=1):
        score, feedback = check(student)
        print(f"Q{i}: {feedback}")
        total += score

    print(f"\nCurrent grade: {total} / {len(checks)}")


# Main stem
if __name__ == "__main__":
    student = import_student_module()
    run_grader(student)
