"""Run the HGB-run PathSnap EDA module from the repository root."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module("hgb_pathsnap.eda", run_name="__main__")
