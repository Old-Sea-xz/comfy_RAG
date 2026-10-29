#!/usr/bin/env python3
"""Compatibility CLI/import for the shared dataset-neutral case executor."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyrouter import case_execution as _execution

if __name__ == "__main__":
    _execution.main()
else:
    # Preserve old helper imports and test patch locations without copying code.
    sys.modules[__name__] = _execution
