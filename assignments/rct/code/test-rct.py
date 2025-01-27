# Imports
import os, sys
import importlib.util
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Run student's script
student_path = os.path.join('.', 'rct.py')
spec = importlib.util.spec_from_file_location('student_model', student_path)
student_module = importlib.util.module_from_spec(spec)
sys.modules['student_model'] = student_module
spec.loader.exec_module(student_module)

# Test 1: declare model
try:
    model = student_module.model
except AttributeError:
    print('Script does not declare a `model` object.')
    exit(1)

# Test 2: Assert type
msg2 = 'Object model should be a RegressionResultsWrapper instance, not {}.'
if not isinstance(model, RegressionResultsWrapper):
    print(msg2.format(type(model)))
    exit(2)

# Test 3: Assert estimates
EXPECTED_PARAM = 0.07997186767333542
try:
    msg3 = "Estimate for 'dark' is {}, expected {}."
    actual_value = model.params['dark'].item()
    assert abs(actual_value - EXPECTED_PARAM) < 1e-6, (
        msg3.format(actual_value, EXPECTED_PARAM)
    )
except AssertionError as e:
    print(f'{e}')
    exit(3)

# Success!
print('All tests passed!')
