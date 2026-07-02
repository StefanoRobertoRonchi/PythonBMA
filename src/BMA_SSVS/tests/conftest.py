import os
import sys

# Ensure the package can be imported from tests even when pytest is run from the project root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
