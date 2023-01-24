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
experiment_name = "2022_12_19_Euglena_3"

current_experiment= DOMEexp.open_experiment(experiment_name, experiments_directory)    

analised_data_path = os.path.join(experiments_directory, experiment_name, 'analysis_data.npz')

# load or extract data
# If the file analyisi_data.npz is not found data are extracted using DOMEtracker.
# Otherwise the data from the existing file are loaded.
if not os.path.isfile(analised_data_path):
    positions, inactivity = DOMEtracker.extract_data_from_images(os.path.join(experiments_directory, experiment_name), AREA_RANGE, COMPAC_RANGE)
    current_experiment.save_data(title="analysis_data", positions=positions, inactivity=inactivity)
    positions = positions.astype(np.float32)
else:  
    with current_experiment.get_data('analysis_data.npz') as data:
        positions = data['positions'].astype(np.float32)
        inactivity = data['inactivity']

positions[positions==-1]=np.nan


positions[inactivity!=0]=np.nan


# Plot trajectories
fig=plt.figure(figsize=(20,20),dpi=72)
img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name), 0)
if len(img.shape)==2:
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

plt.plot(positions[:,:,0],positions[:,:,1],'o-', markersize=3)

for i in range(positions.shape[1]):
    starting_index = (~np.isnan(positions[:,i,0])).argmax(axis=0)
    plt.text(positions[starting_index,i,0], positions[starting_index,i,1], str(i), fontsize = 22, color = DOMEgraphics.std_color_for_index(i))
fig.show()















