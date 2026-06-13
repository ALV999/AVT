"""Allow running: python -m audiobrain"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audiobrain.tui.app import run
run()
