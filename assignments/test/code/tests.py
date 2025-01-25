# Imports
import os, sys
import importlib.util
from linearmodels.panel.results import PanelEffectsResults

# Run student's script
student_path = os.path.join('.', 'test.py')
spec = importlib.util.spec_from_file_location('student_model', student_path)
student_module = importlib.util.module_from_spec(spec)
sys.modules["student_model"] = student_module
spec.loader.exec_module(student_module)

# Test 1: declare model
try:
    model = student_module.model
except AttributeError:
    print('Script does not declare a `model` object.')
    exit(1)

# Test 2: Assert type
msg2 = "Object `model` should be a 'PanelEffectsResult' instance, not {}."
if not isinstance(model, PanelEffectsResults):
    print(msg2.format(type(model)))
    exit(2)

# Test 3: Assert estimates
expected_value = 1.3375465850237276
try:
    msg3 = "Estimate for 'treat' is {}, expected {}."
    actual_value = model.params['treat'].item()
    assert abs(actual_value - expected_value) < 1e-6, (
        msg3.format(actual_value, expected_value)
    )
except AssertionError as e:
    print(f'{e}')
    exit(3)

# Success!
print('All tests passed!')
