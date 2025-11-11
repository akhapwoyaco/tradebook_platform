import sys
from pathlib import Path

# Add the project root directory to the Python path
# This allows modules to be imported correctly during testing.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
