# conftest.py (at project root, same level as src/ and tests/)
import sys
from pathlib import Path

# Make the examples directory importable during tests
sys.path.insert(0, str(Path(__file__).parent))