"""
Pytest configuration - ensures tests/ is discoverable as a package root
relative to the project, so imports resolve consistently regardless of
the directory pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
