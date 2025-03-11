# Imports
import os

# Find student's script (rct-<student ID>.py)
student_files = [
    f for f in os.listdir() if f.startswith('rct-') and f.endswith('.py')
]
if len(student_files) == 0:
    raise FileNotFoundError('Unable to find submission (rct-<student ID>.py).')
elif len(student_files) > 1:
    raise ValueError('More than one submission found!')
student_path = student_files[0]
