"""Shared pytest configuration for the I-ROCKET package."""

import os
import sys
from pathlib import Path

# Avoid nested OpenMP/Numba oversubscription in the test process.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# The current project still ships flat modules. Keep imports explicit until the
# package namespace is addressed in a later, deliberate compatibility change.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
