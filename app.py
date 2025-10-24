"""
Hugging Face Spaces Entry Point
================================

This is the entry point for Hugging Face Spaces deployment.
It redirects to the main dashboard.
"""

import subprocess
import sys

if __name__ == "__main__":
    # Run the main dashboard
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/visualization/dashboards/main_dashboard.py",
        "--server.port=7860",
        "--server.address=0.0.0.0"
    ])

