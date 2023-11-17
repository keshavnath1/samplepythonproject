#!/bin/bash

# Enable verbose mode
set -x

# Run pytest with detailed output
pytest test_temporal_model.py -s

pip install pytest-json-report
pytest --plugins


# Run pytest to generate JSON report
pytest --json-report --json-report-file=cucumber_report.json

# Run the Python script to generate the HTML report
python ./reports/prepare_html_cucumber.py

echo "Automation script executed successfully."

# Disable verbose mode
set +x
