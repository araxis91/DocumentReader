#!/usr/bin/env python3
"""
Standalone GUI launcher for DocumentReader application.
Run this script to launch the Streamlit GUI interface.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Import and launch the GUI
    from src.gui import launch_gui
    launch_gui() 