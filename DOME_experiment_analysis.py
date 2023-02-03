#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created by Andrea

@author: andrea
"""


import cv2
import numpy as np
import scipy
import glob
import matplotlib.pyplot as plt
import os
from typing import List
import DOME_experiment_manager as DOMEexp
import DOME_graphics as DOMEgraphics
import DOME_tracker as DOMEtracker

# MAIN

AREA_RANGE = [100, 600]
COMPAC_RANGE = [0.5, 0.9]

experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
experiment_name = "2023_02_01_EuglenaG_9"

current_experiment= DOMEexp.open_experiment(experiment_name, experiments_directory)    

analised_data_path = os.path.join(experiments_directory, experiment_name, 'analysis_data.npz')

# load or extract data
# If the file analyisi_data.npz is not found data are extracted using DOMEtracker.
# Otherwise the data from the existing file are loaded.
if not os.path.isfile(analised_data_path):
    positions, inactivity = DOMEtracker.extract_data_from_images(os.path.join(experiments_directory, experiment_name), AREA_RANGE, COMPAC_RANGE)
    current_experiment.save_data(title="analysis_data", positions=positions, inactivity=inactivity)
    positions = positions
else:  
    with current_experiment.get_data('analysis_data.npz') as data:
        positions = data['positions']
        inactivity = data['inactivity']

# replace estimated positions with interpolated ones
positions[inactivity!=0]=np.nan
interp_positions = DOMEtracker.interpolate_positions(positions)

img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name), 0)
DOMEgraphics.draw_trajectories(interp_positions, inactivity, img, "trajectories", np.inf)















