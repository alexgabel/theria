import os
import sys
from pathlib import Path


# Ensure project root is on sys.path for test collection
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # Register custom markers to silence PytestUnknownMarkWarning
    config.addinivalue_line("markers", "gpu: tests requiring CUDA")
    config.addinivalue_line("markers", "phase9: Phase 9 correctness suite")
