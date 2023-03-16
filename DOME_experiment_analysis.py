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
import random
from typing import List
import DOME_experiment_manager as DOMEexp
import DOME_graphics as DOMEgraphics
import DOME_tracker as DOMEtracker



def moving_average(x, window, weights=[], axis=0):
    #x=np.array(x).astype(float)
    x=np.ma.filled(x, fill_value=np.nan)
    
    if weights == []:
        weights=np.ones(window)
    else:
        weights=np.array(weights)
    
    kernel=weights/sum(weights)
    y=np.zeros_like(x) * np.nan
    
    for i in range(x.shape[axis-1]):
        #y[:,i]= np.convolve(x[:,i], kernel, 'same')
        data = np.take(x, i, axis-1)
        out_data = data.copy(); out_data
        
        if sum(~np.isnan(data)) >= window:
            values=np.convolve(data, kernel, 'valid'); values
            edge=round(np.floor(window/2))

            (out_data[edge:-edge])[~np.isnan(values)]

            (out_data[edge:-edge])[~np.isnan(values)] = values[~np.isnan(values)]
            
            out_data
            
        if axis==0:
            y[:,i]=out_data
        elif axis==1:
            y[i,:]=out_data
            
    return y

def angle_diff(unit1, unit2):
    angles=np.zeros_like(unit1)
    
    for i in range(unit1.shape[0]):
        for j in range(unit1.shape[1]):
            angle1=unit1[i,j]
            angle2=unit2[i,j]
            
            if not np.isnan(angle1) and not np.isnan(angle2):
                phi = abs(angle2-angle1) % (2 * np.pi)
                sign = 1
                # calculate sign
                if not ((angle1-angle2 >= 0 and angle1-angle2 <= np.pi) or (
                        angle1-angle2 <= - np.pi and angle1-angle2 >= -2 * np.pi)):
                    sign = -1
                if phi > np.pi:
                    result = 2 * np.pi-phi
                else:
                    result = phi
            
                angles[i,j]=result*sign
            else:
                angles[i,j]=np.nan
            
    return angles

def my_histogram(data : np.array, bins, normalize=False):
    #data=np.array(data)
    
    number_of_series=len(data)
    
    values=np.zeros([number_of_series, len(bins)-1])
    
    for i in range(number_of_series):
        val, b=np.histogram(data[i], bins)
    
        if normalize:
            val = val/sum(val)
        
        values[i]=val
        
        positions=bins[:-1] + (bins[1:]-bins[:-1])/(number_of_series+1)*(i+1)
        plt.bar(positions, val, width=0.8*np.diff(bins)/number_of_series)
    
    plt.xticks(bins)
    plt.xlim([min(bins), max(bins)])
        
# MAIN
experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
experiment_name = "2023_02_20_Euglena_4"
output_folder ='tracking1'

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
img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name, 'images'), 60)
DOMEgraphics.draw_trajectories(interp_positions, [], inactivity, img, "trajectories", 1, -1)

# smooth trajectories
#interp_positions = np.ma.array(interp_positions, mask=np.isnan(interp_positions))
interp_positions[:,:,0] = moving_average(interp_positions[:,:,0],3)
interp_positions[:,:,0] = moving_average(interp_positions[:,:,0],3)
interp_positions[:,:,1] = moving_average(interp_positions[:,:,1],3)
interp_positions[:,:,1] = moving_average(interp_positions[:,:,1],3)
DOMEgraphics.draw_trajectories(interp_positions, [], inactivity, img, "smoothed trajectories", 1, -1)


# length of trajectories
lengths = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=0)

# discard short trajectories
min_traj_length = 10
interp_positions[:,lengths<min_traj_length,:]= np.nan

# displacements
#displacements = interp_positions[1:,:,:] - interp_positions[:-1,:,:]
displacements = np.gradient(interp_positions, axis=0)

# speed
speeds = np.linalg.norm(displacements, axis=2)
speeds = np.ma.array(speeds, mask=np.isnan(speeds))

speeds_smooth = moving_average(speeds, 3)
speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))


# accelearation
#acc = speeds_smooth[1:,:] - speeds_smooth[:-1,:]
acc = np.gradient(speeds_smooth, axis=0)
acc = np.ma.array(acc, mask=np.isnan(acc))
acc_smooth = moving_average(acc, 3)
acc_smooth = np.ma.array(acc_smooth, mask=np.isnan(acc_smooth))

# directions
norm_disp = np.divide(displacements,np.stack([speeds,speeds], axis=2)+0.001)
norm_disp = np.ma.array(norm_disp, mask=np.isnan(norm_disp))
directions=np.arctan2(norm_disp[:,:,1],norm_disp[:,:,0])

# compue angular velocity
ang_vel = angle_diff(directions[1:,:], directions[:-1,:])
ang_vel = np.ma.array(ang_vel, mask=np.isnan(ang_vel))

# inergrate angular velocity to obtain continous direction
starting_dir = np.zeros([1, directions.shape[1]])
for i in range(directions.shape[1]):
    starting_idx = np.ma.flatnotmasked_edges(directions[:,i])
    try:
        starting_dir[0,i]= directions[np.ma.flatnotmasked_edges(directions[:,i])[0], i] 
    except:
        pass
directions_reg = starting_dir + np.cumsum(ang_vel, axis=0)
directions_reg[directions_reg.mask==1] = np.nan
directions_reg_smooth = moving_average(directions_reg, 3)

# differentiate continous direction to obtain smooth ang vel
ang_vel_smooth = np.gradient(directions_reg_smooth, axis=0)
#ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
ang_vel_smooth = moving_average(ang_vel_smooth, 3)
ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))


# inputs
inputs = np.mean(np.mean(patterns, axis=1), axis=1)

# average (over time) values for different inputs
speeds_on = speeds_smooth[inputs[:,0]>100,:]
speeds_off = speeds_smooth[inputs[:,0]<100,:]

acc_on =  np.abs(acc_smooth[inputs[:,0]>100,:])
acc_off = np.abs(acc_smooth[inputs[:,0]<100,:])

ang_vel_on  = np.abs(ang_vel_smooth[inputs[:-1,0]>100,:])
ang_vel_off = np.abs(ang_vel_smooth[inputs[:-1,0]<100,:])


# Plots

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

# # directions histogram
# plt.title('directions')
# bins = np.linspace(-np.pi, np.pi, 9)
# plt.hist(directions.flatten(), bins)
# plt.xlabel('Direction [rad]')
# plt.xticks(bins)
# plt.gca().set_xlim([-np.pi, np.pi])
# plt.ylabel('Count')
# plt.grid()
# plt.show()

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
#plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.ma.median(speeds,axis=1))
plt.plot(np.linspace(0, time_intants-1, time_intants),np.ma.median(speeds_smooth,axis=1))
plt.fill_between(np.linspace(0, time_intants-1, time_intants), np.min(speeds_smooth,axis=1), np.max(speeds_smooth,axis=1),alpha=0.5)
plt.xlabel('Time [frames]')
plt.ylabel('Speed [px/frame]')
plt.gca().set_xlim([0, time_intants-1])
plt.gca().set_ylim(0)
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(3, 1, 2)
#plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.ma.median(np.abs(acc),axis=1))
plt.plot(np.linspace(0, time_intants-1, time_intants),np.ma.median(np.abs(acc_smooth),axis=1))
plt.fill_between(np.linspace(0, time_intants-1, time_intants), np.min(np.abs(acc_smooth),axis=1), np.max(np.abs(acc_smooth),axis=1),alpha=0.5)
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Acc [px/frame^2]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(3, 1, 3)
#plt.plot(np.linspace(0, time_intants-3, time_intants-2),np.ma.median(np.abs(ang_vel),axis=1))
plt.plot(np.linspace(0, time_intants-2, time_intants-1),np.ma.median(np.abs(ang_vel_smooth),axis=1))
plt.fill_between(np.linspace(0, time_intants-2, time_intants-1), np.min(np.abs(ang_vel_smooth),axis=1), np.max(np.abs(ang_vel_smooth),axis=1),alpha=0.5)
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Ang Vel [rad/frame]')
plt.xlabel('Time steps [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# boxplots
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on, axis=0), np.mean(speeds_off, axis=0), (np.mean(speeds_on, axis=0) - np.mean(speeds_off, axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Speed [px/frame]')
plt.title('Boxplots')
plt.subplot(3, 1, 2)
data_to_plot = list(map(lambda X: [x for x in X if x],[np.mean(acc_on, axis=0), np.mean(acc_off, axis=0), (np.mean(acc_on, axis=0) - np.mean(acc_off, axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Acc [px/frame^2]')
plt.subplot(3, 1, 3)
data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on, axis=0), np.mean(ang_vel_off, axis=0), (np.mean(ang_vel_on, axis=0) - np.mean(ang_vel_off, axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Ang Vel [rad/frame]')
plt.show()

# focused boxplots
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on[:5,:], axis=0), np.mean(speeds_off[-5:,:], axis=0), (np.mean(speeds_on[:5,:], axis=0) - np.mean(speeds_off[-5:,:], axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Speed [px/frame]')
plt.title('Boxplots: just before and after the switch')
plt.subplot(3, 1, 2)
data_to_plot = list(map(lambda X: [x for x in X if x],[np.mean(acc_on[:5,:], axis=0), np.mean(acc_off[-5:,:], axis=0), (np.mean(acc_on[:5,:], axis=0) - np.mean(acc_off[-5:,:], axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Acc [px/frame^2]')
plt.subplot(3, 1, 3)
data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on[:5,:], axis=0), np.mean(ang_vel_off[-5:,:], axis=0), (np.mean(ang_vel_on[:5,:], axis=0) - np.mean(ang_vel_off[-5:,:], axis=0))]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
plt.axhline(0, color='gray')
plt.ylabel('Ang Vel [rad/frame]')
plt.show()

# histograms
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
plt.title('Histograms')
bins=np.linspace(0, 40, round(40/5+1))
my_histogram([np.mean(speeds_on, axis=0).compressed(), np.mean(speeds_off, axis=0).compressed()], bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Speed [px/frame]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.subplot(3, 1, 2)
bins=np.linspace(0, 5, round(10+1))
my_histogram([np.mean(acc_on, axis=0).compressed(), np.mean(acc_off, axis=0).compressed()], bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Acc [px/frame^2]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.subplot(3, 1, 3)
bins=np.linspace(0, 1, round(10+1))
my_histogram([np.mean(ang_vel_on, axis=0).compressed(), np.mean(ang_vel_off, axis=0).compressed()], bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Ang Vel [rad/frame]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.show()


# focused histograms
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
plt.title('Histograms: just before and after the switch')
bins=np.linspace(0, 40, round(40/5+1))
my_histogram([np.mean(speeds_on[:5,:], axis=0).compressed(), np.mean(speeds_off[-5:,:], axis=0).compressed()] , bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Speed [px/frame]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.subplot(3, 1, 2)
bins=np.linspace(0, 5, round(10+1))
my_histogram([np.mean(acc_on[:5,:], axis=0).compressed(), np.mean(acc_off[-5:,:], axis=0).compressed()] , bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Acc [px/frame^2]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.subplot(3, 1, 3)
bins=np.linspace(0, 1, round(10+1))
my_histogram([np.mean(ang_vel_on[:5,:], axis=0).compressed(), np.mean(ang_vel_off[-5:,:], axis=0).compressed()] , bins, True)
plt.legend(labels=['Light ON', 'Light OFF'])
plt.xlabel('Ang Vel [rad/frame]')
#plt.gca().set_ylim([0, 0.25])
plt.ylabel('Agents')
plt.grid()
plt.show()

# Select one agent
agent=np.argmax(lengths)
#agent=random.choice(np.arange(len(lengths))[lengths >= min_traj_length])

# Speed and Acceleration of one agent
plt.figure(figsize=(9,6))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, time_intants-1, time_intants),speeds_smooth[:,agent])
#plt.plot(np.linspace(0, time_intants-1, time_intants),speeds[:,agent], '--')
plt.title('Movement of agent '+str(agent))
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Speed [px/frame]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, time_intants-1, time_intants),np.abs(acc_smooth[:,agent]))
#plt.plot(np.linspace(0, time_intants-1, time_intants),np.abs(acc[:,agent]),'--')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Abs Acc [px/frame^2]')
plt.xlabel('Time [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# Direction and Angular Velocity of one agent
plt.figure(figsize=(9,6))
plt.subplot(2, 1, 1)
#plt.plot(np.linspace(1, time_intants-1, time_intants-1),directions[:,agent])
plt.plot(np.linspace(1, time_intants-1, time_intants-1),directions_reg_smooth[:,agent])
#plt.plot(np.linspace(1, time_intants-1, time_intants-1),directions_reg[:,agent],'--')
plt.title('Movement of agent '+str(agent))
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Direction [rad]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)

# plt.subplot(3, 1, 2)
# plt.plot(np.linspace(0, time_intants-2, time_intants-1),directions[:,agent])
# plt.plot(np.linspace(0, time_intants-2, time_intants-1),directions_smooth[:,agent])
# plt.title('Movement of agent '+str(agent))
# plt.gca().set_xlim([0, time_intants-1])
# plt.gca().set_ylim([-np.pi, np.pi])
# plt.ylabel('Direction [rad]')
# plt.yticks(np.linspace(-np.pi, np.pi, 5))
# plt.grid()
# DOMEgraphics.highligth_inputs(inputs)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(1, time_intants-1, time_intants-1),np.abs(ang_vel_smooth[:,agent]))
#plt.plot(np.linspace(1, time_intants-1, time_intants-1),np.abs(ang_vel[:,agent]),'--')
plt.gca().set_xlim([0, time_intants-1])
plt.ylabel('Abs Angular Vel [rad/frame]')
plt.xlabel('Time [frames]')
plt.grid()
DOMEgraphics.highligth_inputs(inputs)
plt.show()

# Barplots of one agent
plt.figure(figsize=(4,6))
plt.subplot(3, 1, 1)
data_to_plot = list(map(lambda X: [x for x in X if x], [speeds_on[:,agent], speeds_off[:,agent]]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
plt.ylabel('Speed [px/frame]')
plt.title('Average values of agent '+ str(agent))
plt.subplot(3, 1, 2)
data_to_plot = list(map(lambda X: [x for x in X if x], [acc_on[:,agent], acc_off[:,agent]]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
plt.ylabel('Acc [px/frame^2]')
plt.subplot(3, 1, 3)
data_to_plot = list(map(lambda X: [x for x in X if x], [ang_vel_on[:,agent], ang_vel_off[:,agent]]))
plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
plt.ylabel('Ang Vel [rad/frame]')
plt.show()

# plot trajectory of one agent
img = DOMEgraphics.get_img_at_time(os.path.join(experiments_directory, experiment_name, 'images'), 60)
DOMEgraphics.draw_trajectories(interp_positions[:,agent:agent+1,:], [], inactivity[:,agent:agent+1], img, "trajectory of agent " +str(agent), np.inf, time_window=-1);





