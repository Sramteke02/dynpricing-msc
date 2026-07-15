"""Make the ``src`` layout importable when running tests without installation."""

import sys
from pathlib import Path

src = Path(__file__).parent / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
