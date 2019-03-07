[TOC]



# OVERVIEW

This repository contains multiple scripts demonstrating different functionality with tensorflow. The scripts are written in python and use the KERAS library to call and run tensorflow.



All scripts within this repository generate their own datasets.



# DESCRIPTIONS



## keras-multi-object-detection.py

### Description

This script uses keras to find the centre point of a variable number of black objects on a white background. It starts by creating a dataset of randomly generated images and then trains a keras model to detect the centre points.



### Important notes

For multiple object detection the output needs to be of a defined length. So it fails if there are variable numbers of objects. To fix this the output can be padded out with zeros. The zeros are replaced with xy coordinates as they are detected.

If a random set of xy coordinates are generated as the output for the training set then the coordinates need to be sorted. Otherwise when multiple coordinates are present in a single image it trains the NN to detect a position halfway between the two points.