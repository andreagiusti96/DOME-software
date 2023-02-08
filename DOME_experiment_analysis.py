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


def highligth_inputs(inputs):
    ons =np.where(inputs[:-2,0]-inputs[1:-1,0]==200)[0]
    offs =np.where(inputs[:-2,0]-inputs[1:-1,0]==-200)[0]
    for i in range(len(ons)):
        plt.axvspan(ons[i], offs[i], color='red', alpha=0.5)


# MAIN

AREA_RANGE = [100, 600]
COMPAC_RANGE = [0.5, 0.9]

experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
experiment_name = "2023_02_01_EuglenaG_9"
output_folder ='tracking2'

current_experiment= DOMEexp.open_experiment(experiment_name, experiments_directory)    

analised_data_path = os.path.join(experiments_directory, experiment_name, output_folder, 'analysis_data.npz')

# load data
if not os.path.isfile(analised_data_path):
    print(f'File {analised_data_path} not found.\nFirst execute tracking with DOME_tracker.')
    exit()

with current_experiment.get_data('data.npz') as data:
    patterns = data['patterns']
    activation_times = data['activation_times']
    
with current_experiment.get_data(os.path.join(output_folder,'analysis_data.npz')) as data:
    positions = data['positions']
    inactivity = data['inactivity']


time_intants = positions.shape[0]
number_of_agents = positions.shape[1]

# replace estimated positions with interpolated ones
positions[inactivity!=0]=np.nan
interp_positions = DOMEtracker.interpolate_positions(positions)

img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name), 60)
DOMEgraphics.draw_trajectories(interp_positions, [], inactivity, img, "trajectories", np.inf)


#interp_positions = np.ma.array(interp_positions, mask=np.isnan(interp_positions))

# length of trajectories
length = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=0)


displacements = np.zeros([time_intants-1, number_of_agents, 2])
for i in range(0,120):
    displacements[i,:,:] = interp_positions[i+1,:,:] - interp_positions[i,:,:]


# speed
speeds = np.linalg.norm(displacements, axis=2)
speeds = np.ma.array(speeds, mask=np.isnan(speeds))

# directions
norm_disp = np.divide(displacements,np.stack([speeds,speeds], axis=2))
norm_disp = np.ma.array(norm_disp, mask=np.isnan(norm_disp))
directions=np.arctan2(norm_disp[:,:,0],norm_disp[:,:,1])

# inputs
inputs = np.mean(np.mean(patterns, axis=1), axis=1)



# Plots


# length of trajectories histogram
plt.title('Trajectories duration')
bins = [0,5,10,20,40,60,80,100, 120]
plt.hist(length, bins)
plt.xlabel('Time steps [frames]')
plt.xticks(bins)
plt.gca().set_xlim([0, 120])
plt.ylabel('Count')
plt.grid()
plt.show()

# directions histogram
plt.title('directions')
bins = np.linspace(-np.pi, np.pi, 9)
plt.hist(directions[10,:], bins)
plt.xlabel('Direction [rad]')
plt.xticks(bins)
plt.gca().set_xlim([-np.pi, np.pi])
plt.ylabel('Count')
plt.grid()
plt.show()

# Inputs
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,0], color='blue')
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,1], color='green')
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,2], color='red')
plt.title('Inputs')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Brightness')
plt.grid()
plt.show()

# Speed
plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.mean(speeds,axis=1))
plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.min(speeds,axis=1))
plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.max(speeds,axis=1))
plt.title('Average speed over time')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('speed [px/frame]')
plt.grid()
plt.show()

# Speed and Direction of one agent
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, time_intants-2, time_intants-1),speeds[:,1])
plt.title('Speed over time')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Speed [px/frame]')
plt.grid()
highligth_inputs(inputs)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, time_intants-2, time_intants-1),directions[:,1])
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.gca().set_ylim([-np.pi, np.pi])
plt.ylabel('Direction [rad]')
plt.yticks(np.linspace(-np.pi, np.pi, 5))
plt.grid()
highligth_inputs(inputs)
plt.show()

# number of agents
agents_number = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=1)
plt.plot(np.linspace(0, time_intants-1, time_intants),agents_number)
plt.title('Number of detected agents over time')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Count')
plt.grid()
plt.show()





