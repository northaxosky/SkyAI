"""Pytest Configuration: shared fixtures and environment setup"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default HF Cache to a repo-local directory
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache"))