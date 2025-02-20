# Imports
import os, sys
import importlib.util
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Find student's script (rct-<student ID>.py)
student_files = [
    f for f in os.listdir() if f.startswith('rct-') and f.endswith('.py')
]
if len(student_files) == 0:
    raise FileNotFoundError('Unable to find submission (rct-<student ID>.py).')
elif len(student_files) > 1:
    raise ValueError('More than one submission found!')
student_path = student_files[0]

# Run student's script
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
EXPECTED_PARAM = 0.0866918272939628
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
