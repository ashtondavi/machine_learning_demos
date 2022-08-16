
# Overview

This repository contains a python package for several machine learning demo 
pipelines including image classification, object detection, and similarity 
models

# Virtual environment setups (python 3.7 and latest)
python -3.7-64 -m venv 3.7_env -> not working?
python -m venv env

# Usage

To run the pipelines install the packages from the repos base directory by
by running "python setup.py install" in either the windows cmd or vscode 
powershell.

## Commands to run pipelines from windows cmd or vscode

python demos/pipelines/object_detection.py --folder "./example_data/images"
python demos/pipelines/classification.py --output "./classifier" --epochs "10" 