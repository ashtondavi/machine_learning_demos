
# Overview

This repository is a playpen area to work on some machine learning pipeline 
demos for image classification and bounding box detection.

# Usage (from vscode)

Set up a virtual environment by running "python -m venv .env"

Switch the the virtual environment

Install the demos packages from the repos base directory by running "pip install 
-e ."

copy and paste one of the commands below inot the powershell to run with 
example data


## Commands to run pipelines from windows cmd or vscode

python demos/pipelines/bbox_detection.py --folder "./example_data/images"
python demos/pipelines/classification.py --output "./classifier" --epochs "10" 