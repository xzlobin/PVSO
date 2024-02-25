import numpy as np
import cv2
import yaml
import os

script_folder = os.path.dirname(os.path.realpath(__file__))

with open(f'{script_folder}/camera_parameters.yaml', 'r') as f:
    camera_parsed = yaml.load(f)
