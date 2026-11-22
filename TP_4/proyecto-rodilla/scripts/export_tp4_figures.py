"""Compatibility wrapper for the current static report exports."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'scripts/build_demo.py')],check=True)
subprocess.run([sys.executable,str(root/'scripts/build_site.py')],check=True)
