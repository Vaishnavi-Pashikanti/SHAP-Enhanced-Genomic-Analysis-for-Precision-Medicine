"""
Wrapper to run tests from the Complete_precision_medcine directory
while setting the proper working directory and paths.
"""

import os
import sys

# Change to parent directory (project root)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)

# Now add current directory (Complete_precision_medcine) to path
current_dir = os.path.join(parent_dir, 'Complete_precision_medcine')
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import and run tests
from test_treatment_benefit_system import run_all_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
