#!/bin/bash
set -e
pip install flask scikit-learn pandas numpy gunicorn
python3 data/generate_data.py
python3 analysis/engine.py
echo "Build complete."
