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



def moving_average(x, w):
    kernel=np.ones(w)/w
    y=np.zeros_like(x)
    for agent in range(x.shape[1]):
        y[:,agent]= np.convolve(x[:,agent], kernel, 'same')
    return y

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

# plot trajectories
img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name), 60)
DOMEgraphics.draw_trajectories(interp_positions, [], inactivity, img, "trajectories", np.inf)


# length of trajectories
lengths = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=0)

# discard short trajectories
min_traj_length = 10
interp_positions[:,lengths<min_traj_length,:]= np.nan

# displacements
displacements = interp_positions[1:,:,:] - interp_positions[:-1,:,:]

# speed
speeds = np.linalg.norm(displacements, axis=2)
speeds = np.ma.array(speeds, mask=np.isnan(speeds))

speeds_smooth = moving_average(speeds, 3)
speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))

# directions
norm_disp = np.divide(displacements,np.stack([speeds,speeds], axis=2))
norm_disp = np.ma.array(norm_disp, mask=np.isnan(norm_disp))
directions=np.arctan2(-norm_disp[:,:,1],norm_disp[:,:,0])

directions_smooth = moving_average(directions, 3)

# accelearation
acc = speeds_smooth[1:,:] - speeds_smooth[:-1,:]
acc = np.ma.array(acc, mask=np.isnan(acc))
acc_smooth = moving_average(acc, 3)
acc_smooth = np.ma.array(acc_smooth, mask=np.isnan(acc_smooth))

ang_vel = directions_smooth[1:,:] - directions_smooth[:-1,:]
ang_vel = np.ma.array(ang_vel, mask=np.isnan(ang_vel))
ang_vel_smooth = moving_average(ang_vel, 3)
ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))


# inputs
inputs = np.mean(np.mean(patterns, axis=1), axis=1)

# average (over time) values for different inputs
speeds_on = np.mean(speeds[inputs[:-1,0]>100,:], axis=1)
speeds_off = np.mean(speeds[inputs[:-1,0]<100,:], axis=1)

acc_on =  np.mean(np.abs(acc[inputs[:-2,0]>100,:]), axis=1)
acc_off = np.mean(np.abs(acc[inputs[:-2,0]<100,:]), axis=1)

ang_vel_on  = np.mean(np.abs(ang_vel[inputs[:-2,0]>100,:]), axis=1)
ang_vel_off = np.mean(np.abs(ang_vel[inputs[:-2,0]<100,:]), axis=1)


# Plots


# length of trajectories histogram
plt.title('Trajectories duration')
bins = [0,5,10,20,40,60,80,100, 120]
plt.hist(lengths, bins)
plt.axvline(min_traj_length, color='red')
plt.xlabel('Time steps [frames]')
plt.xticks(bins)
plt.gca().set_xlim([0, 120])
plt.ylabel('Count')
plt.grid()
plt.show()

# directions histogram
plt.title('directions')
bins = np.linspace(-np.pi, np.pi, 9)
plt.hist(directions.flatten(), bins)
plt.xlabel('Direction [rad]')
plt.xticks(bins)
plt.gca().set_xlim([-np.pi, np.pi])
plt.ylabel('Count')
plt.grid()
plt.show()

# Inputs
plt.figure(figsize=(9,6))
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,0], color='blue')
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,1], color='green')
plt.plot(np.linspace(0, time_intants-1, time_intants),inputs[:,2], color='red')
plt.title('Inputs')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Brightness')
plt.grid()
plt.show()

# Average Speed, Acc and Angular Velocity
plt.figure(figsize=(9,6))
plt.subplot(3, 1, 1)
plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.mean(speeds,axis=1))
#plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.mean(speeds_smooth,axis=1))
plt.fill_between(np.linspace(0, time_intants-2, time_intants-1), np.min(speeds,axis=1), np.max(speeds,axis=1),alpha=0.5)
plt.xlabel('Time [frames]')
plt.ylabel('Speed [px/frame]')
plt.gca().set_xlim([0, time_intants-1])
plt.gca().set_ylim(0)
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(3, 1, 2)
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.mean(np.abs(acc),axis=1))
#plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.mean(np.abs(acc_smooth),axis=1))
plt.fill_between(np.linspace(0, time_intants-3, time_intants-2), np.min(np.abs(acc),axis=1), np.max(np.abs(acc),axis=1),alpha=0.5)
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Acc [px/frame^2]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(3, 1, 3)
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.mean(np.abs(ang_vel),axis=1))
#plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.mean(np.abs(ang_vel_smooth),axis=1))
plt.fill_between(np.linspace(0, time_intants-3, time_intants-2), np.min(np.abs(ang_vel),axis=1), np.max(np.abs(ang_vel),axis=1),alpha=0.5)
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Ang Vel [rad/frame]')
plt.xlabel('Time steps [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# barplots
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
plt.boxplot([speeds_on, speeds_off],labels=['Light ON', 'Light OFF'])
plt.ylabel('Speed [px/frame]')
plt.subplot(3, 1, 2)
plt.boxplot([acc_on, acc_off],labels=['Light ON', 'Light OFF'])
plt.ylabel('Acc [px/frame^2]')
plt.subplot(3, 1, 3)
plt.boxplot([ang_vel_on, ang_vel_off],labels=['Light ON', 'Light OFF'])
plt.ylabel('Ang Vel [rad/frame]')
plt.show()

# Speed and Acceleration of one agent
agent=1
plt.figure(figsize=(9,6))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, time_intants-2, time_intants-1),speeds[:,agent])
plt.plot(np.linspace(0, time_intants-2, time_intants-1),speeds_smooth[:,agent])
plt.title('Movement of agent '+str(agent))
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Speed [px/frame]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.abs(acc[:,agent]))
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.abs(acc_smooth[:,agent]))
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Acc [px/frame^2]')
plt.xlabel('Time [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# Direction and Angular Velocity of one agent
plt.figure(figsize=(9,6))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, time_intants-2, time_intants-1),directions[:,agent])
plt.plot(np.linspace(0, time_intants-2, time_intants-1),directions_smooth[:,agent])
plt.title('Movement of agent '+str(agent))
plt.gca().set_xlim([0, time_intants-1])
plt.gca().set_ylim([-np.pi, np.pi])
plt.ylabel('Direction [rad]')
plt.yticks(np.linspace(-np.pi, np.pi, 5))
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.abs(ang_vel[:,agent]))
plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.abs(ang_vel_smooth[:,agent]))
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Angular Vel [rad/frame]')
plt.xlabel('Time [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# number of agents
plt.figure(figsize=(9,3))
agents_number = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=1)
plt.plot(np.linspace(0, time_intants-1, time_intants),agents_number)
plt.title('Number of detected agents over time')
plt.xlabel('Time [frames]')
plt.gca().set_xlim([0, time_intants-1])
plt.gca().set_ylim(0)
plt.ylabel('Count')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()





