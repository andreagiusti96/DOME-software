#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code reads in figures from a folder and for each frame, performs basic image analysis to try and detect agents using contour detection. 
The "agents" detected are then run though an agent matching algorithm to try and figure out if they correspond with an agent in a previous frame
Displacement is then calculated by comaparing the position of the agent in the new frame to the previous frame.
Time between frames is also calcuated by reading the filename, which is timestamped.
A csv file is output with a list of agents with unique agent numbers, and for each frame where they are detected a data point is stored with (displacement, time, frame)

Created by Andrea

@author: andrea
"""


import cv2
import numpy as np
import scipy
import glob
import re
from typing import List
import os
import matplotlib.pyplot as plt
import random

import DOME_graphics as DOMEgraphics
import DOME_experiment_manager as DOMEexp

# old parameters without scaling on Typical_D
# NEW_ID_COST_MIN = 1000
# NEW_ID_COST_DIST_CAP = 100
# DISTANCE_COST_FACTORS = [0, 2]
# INACTIVITY_COST_FACTORS = [0, 1000]

NEW_ID_COST_MIN = 1
NEW_ID_COST_DIST_CAP = 3
DISTANCE_COST_FACTORS = [0, 1]
INACTIVITY_COST_FACTORS = [0, 1]

def matchingCost(distance, inactivity):
    cost = (distance*DISTANCE_COST_FACTORS[0] + distance**2*DISTANCE_COST_FACTORS[1])/(inactivity**2*0.25+1)
    cost += inactivity * INACTIVITY_COST_FACTORS[0] +inactivity**2 * INACTIVITY_COST_FACTORS[1]
    return cost

def plotCosts():
    fig = plt.figure(1,figsize=(19.20,10.80),dpi=100)
    fig.subplots_adjust(top=1.0-0.05, bottom=0.05, right=1.0-0.05, left=0.05, hspace=0, wspace=0) 
    plt.title('Matching cost')
    
    maxdist=4
    distances=np.linspace(0, maxdist)
    inactivity=np.array([0, 1, 2, 3, 4, 5])
    
    matching_cost=np.zeros([len(distances), len(inactivity)])
    
    for i in range(len(inactivity)):
        matching_cost[:,i] = matchingCost(distances, inactivity[i])
        
    new_id_cost_max=(NEW_ID_COST_DIST_CAP**2) + NEW_ID_COST_MIN
    
    plt.plot(distances, matching_cost)
    plt.plot([0, maxdist], NEW_ID_COST_MIN * np.array([1, 1]))
    plt.plot([0, maxdist],  new_id_cost_max* np.array([1, 1]))
    plt.legend( inactivity)
    plt.gca().set_ylim([0, new_id_cost_max*1.5])
    plt.gca().set_xlim([0, maxdist])
    plt.xlabel('distance/TYPICAL_D')


def agentMatching(new_positions : np.array, positions : np.array, inactivity : List):
    """
    Track the objects in subsequent time instants assigning IDs.
    The IDs assignment is formulated as a linear optimization problem and solved with the Hungarian method.
    New IDs can be allocated.
    
    Parameters
    ----------
    new_positions : np.array (Shape=Nx2)
        Positions of detected objects.
    positions : np.array
        Positions of previously detected objects. (Shape=Mx2)
    inactivity : List
        Inactivity counters of the objects. (Shape=M)

    Returns
    -------
    new_ids : List
        IDs assigned to the detected positions. (Shape=N)

    """
    new_positions=np.array(new_positions)
    number_of_objects = sum(DOMEgraphics.valid_positions(positions))
    costs_matching = np.ndarray([len(new_positions), number_of_objects])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])
    
    distances = np.squeeze(scipy.spatial.distance.cdist(new_positions, positions))
    distances = distances/TYPICAL_D
    
    # build the matrix of costs
    for i in range(positions.shape[0]):
        # distances = np.squeeze(scipy.spatial.distance.cdist(new_positions, positions[i,:]))
        # costs_matching[i,:] = distances[i,:]*DISTANCE_COST_FACTORS[0] +distances[i,:]**2*DISTANCE_COST_FACTORS[1]
        # inactivity_cost = (np.array(inactivity)) * INACTIVITY_COST_FACTORS[0] + (np.array(inactivity)**2) * INACTIVITY_COST_FACTORS[1]
        # costs_matching[i,:] += inactivity_cost
        costs_matching[:,i] = matchingCost(distances[:,i], inactivity[i])
        
    for i in range(new_positions.shape[0]):
        cost_newid = np.min([DOMEgraphics.distance_from_edges(new_positions[i])/TYPICAL_D, NEW_ID_COST_DIST_CAP])**2 + NEW_ID_COST_MIN
        costs_newid[i,:] = np.ones([len(new_positions)]) * cost_newid
        
    costs = np.concatenate((costs_matching, costs_newid), axis=1)
    
    # Hungarian optimization algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    cost = costs[row_ind, col_ind].sum()
    
    # update ids
    new_ids = [i for i in col_ind]
    
    print('matching cost = ' + str(round(cost,2)) + '\t avg = ' + str(round(cost/(len(new_ids)+0.001),2)))
    
    return new_ids

def estimate_velocities(positions : np.array):
    """
    Given the past positions of the objects estimates their velocities.

    Parameters
    ----------
    positions : np.array
        Past positions of the objects. Shape=(MxNx2)
        If M<2 all velocities are [0, 0].
        Non valid position are discarded.
    Returns
    -------
    velocities : np.array
        Velocities of the objects. Shape=(Nx2)

    """
    assert len(positions.shape) == 3
    assert positions.shape[2] == 2

    velocities = np.zeros(positions.shape[1:3])
    
    if positions.shape[0] >=2:
        valid_pos_idx = DOMEgraphics.valid_positions(positions[-2])
        velocities[valid_pos_idx] = positions[-1, valid_pos_idx] - positions[-2, valid_pos_idx]
    
    speeds = np.linalg.norm(velocities, axis=1)
    
    #print("avg speed = " + str(round(np.mean(speeds),1)) + "\tmax = " + str(round(max(speeds),1)) + "\tid =" + str(np.argmax(speeds)))
    
    assert velocities.shape[1] == 2
    return velocities

def estimate_positions(old_pos : np.array, velocity : np.array):
    """
    Given the current positions and velocities returns the future estimated positions of objects.
    Positions are validated to be in the range [0, 1920][0, 1080]

    Parameters
    ----------
    old_pos : np.array
        Last positions of the objects. Shape=(Nx2)
    velocity : np.array
        Velocities of the objects. Shape=(Nx2)

    Returns
    -------
    estimated_pos : np.array
        Next positions of the objects. Shape=(Nx2).

    """
    assert len(old_pos.shape) == 2
    assert len(velocity.shape) == 2
    assert old_pos.shape[1] == 2
    
    inertia = 0.66
    
    estimated_pos = old_pos + velocity * inertia
    
    non_valid_pos_idx = ~ DOMEgraphics.valid_positions(estimated_pos)
    estimated_pos[non_valid_pos_idx] = estimated_pos[non_valid_pos_idx]  - velocity[non_valid_pos_idx]* inertia 
    
    return estimated_pos

def interpolate_positions(positions : np.array):
    interpolated_pos=positions.copy()
    
    for obj in range(positions.shape[1]):
        first_index = (~np.isnan(positions[:,obj,0])).argmax(0)
        last_index = positions.shape[0] - (~np.isnan(positions[:,obj,0]))[::-1].argmax(0) -1
        
        nans=np.isnan(positions[first_index:last_index+1,obj,0])
        missing_points = np.where(nans)[0] +first_index
        valid_points = np.where(~nans)[0] +first_index
        
        if len(missing_points) >0 :
            trajectory_x=positions[valid_points,obj,0]
            trajectory_y=positions[valid_points,obj,1]
            interpolated_pos[missing_points,obj,0] = np.interp(missing_points, valid_points, trajectory_x)
            interpolated_pos[missing_points,obj,1] = np.interp(missing_points, valid_points, trajectory_y)
    
        #print(np.concatenate([positions[:last_index+2,obj], interpolated_pos[:last_index+2,obj]], axis=1))
    return interpolated_pos


def test_detection_parameters(fileLocation, bright_thresh, area_r : List, compactness_r : List):
    files = glob.glob(fileLocation +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    filename = random.choice(files)
    img = cv2.imread(filename)

    background = DOMEgraphics.build_background(fileLocation, 25)

    new_contours = DOMEgraphics.get_contours(img, bright_thresh, area_r, compactness_r, background, 0, True)

def extract_data_from_images(fileLocation, bright_thresh : List, area_r : List, compactness_r : List, output_folder : str):
    
    print("Building the background model...")
    background = DOMEgraphics.build_background(fileLocation, 25)

    files = glob.glob(fileLocation +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    frames_number = len(files)
    number_of_objects=0
    n_detected_objects=0

    contours=[];
    positions= np.empty([frames_number, 0, 2], dtype=float)*np.nan;
    inactivity= - np.ones([frames_number, 0], dtype=int );
    
    print("Performing detection and tracking...")
    for counter in range(len(files)):
    #for counter in range(10):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = DOMEgraphics.get_time_from_title(filename)
        print('t = ' + str(time))
        
        # collect contours and positions from new image
        plot_detection_steps = counter == 0
        new_contours = DOMEgraphics.get_contours(img, bright_thresh, area_r, compactness_r, background, n_detected_objects, plot_detection_steps)
        new_positions = DOMEgraphics.get_positions(new_contours)
        n_detected_objects=len(new_positions)
        
        # on first iteration assign new susequent ids to all agents
        if counter == 0: 
            new_ids = list(range(0, n_detected_objects))
        
        # on following iterations perform tracking
        else:
            est_positions=positions[counter]                  # select positions at previous time instant
            est_positions=est_positions[DOMEgraphics.valid_positions(est_positions)] # select valid positions
            new_ids = agentMatching(new_positions, est_positions, inactivity[counter-1])
        
        # discern new and lost objects
        newly_allocated_ids = [i for i in new_ids if i not in range(number_of_objects)]
        lost_obj_ids = [i for i in range(number_of_objects) if i not in new_ids]
        
        # update data
        for new_id in new_ids:
            # for already detected objects update data
            if new_id < number_of_objects:
                positions[counter, new_id] = new_positions[new_ids.index(new_id)]
                contours[new_id] = new_contours[new_ids.index(new_id)]
                inactivity[counter, new_id] = 0
                
            # for new objects allocate new data
            else:
                empty_row= - np.empty([frames_number, 1, 2], dtype=float)*np.nan
                positions = np.concatenate([positions,  empty_row], axis=1)
                positions[counter, number_of_objects] = new_positions[new_ids.index(new_id)]
                
                empty_row= - np.ones([frames_number, 1], dtype=int)
                inactivity = np.concatenate([inactivity,  empty_row], axis=1)
                inactivity[counter, number_of_objects] = 0

                contours.append(new_contours[new_ids.index(new_id)])
                number_of_objects += 1
        
        # for lost objects estimate position and increase inactivity
        for lost_id in lost_obj_ids:
            inactivity[counter, lost_id] = inactivity[counter-1, lost_id] + 1
        
        # estimate velocities and future positions
        up_to_now_positions=positions[0:counter+1]             # select positions up to current time instant
        velocities = estimate_velocities(up_to_now_positions)
        if counter < frames_number-1:
            positions[counter+1] = estimate_positions(positions[counter], velocities)
        
        # check data integrity
        assert all(DOMEgraphics.valid_positions(positions[counter]))
        
        # print image
        fig = DOMEgraphics.draw_trajectories(positions[:counter+1], [], inactivity[:counter+1], img, title='time='+str(time), max_inactivity=3, time_window=5)
        fig.savefig(os.path.join(fileLocation, output_folder,'trk_' + '%04.1f' % time + '.jpeg'), dpi=100)
        
        # print info
        print('total number of objects = '+ str( number_of_objects) )
        print('detected objects = '+ str( n_detected_objects) )
        print('new ids = ' + str(newly_allocated_ids) + '\t tot = '+ str( len(newly_allocated_ids)) )
        print('total lost ids = '+ str( len(lost_obj_ids)) + '\n')
    
    return positions, inactivity

# MAIN
if __name__ == '__main__':
    
    # # Euglena
    # AREA_RANGE = [250, 3000]; COMPAC_RANGE = [0.6, 0.9]; BRIGHT_THRESH = [85]
    # TYPICAL_D = 25
    
    # P. Caudatum
    #AREA_RANGE = [250, 3000]; COMPAC_RANGE = [0.5, 0.9]; BRIGHT_THRESH = [70]
    #TYPICAL_D = 50
    
    # Volvox
    AREA_RANGE = [1000, 6000]; COMPAC_RANGE = [0.7, 1.0]; BRIGHT_THRESH = [70]
    TYPICAL_D = 25
    
    experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
    experiment_name = "2023_06_08_Volvox_1"
    output_folder ='tracking'
    
    current_experiment= DOMEexp.open_experiment(experiment_name, experiments_directory)    
    
    if os.path.isdir(os.path.join(experiments_directory, experiment_name,'images')):
        images_folder=os.path.join(experiments_directory, experiment_name,'images')
    else:
        images_folder=os.path.join(experiments_directory, experiment_name)
    
    test_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)
    
    output_dir = os.path.join(experiments_directory, experiment_name, output_folder)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    
    # extract data
    positions, inactivity = extract_data_from_images(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE, output_dir)
    
    # make video from images
    DOMEgraphics.make_video(output_dir, title='tracking.mp4', fps=2)
    
    # If the file analysis_data.npz is not already existing data are saved
    analised_data_path = os.path.join(output_dir, 'analysis_data.npz')
    if not os.path.isfile(analised_data_path):
        current_experiment.save_data(os.path.join(output_folder, 'analysis_data'), positions=positions, inactivity=inactivity)
    else:
        print(f'The file {analised_data_path} already exists. Data cannot be saved.')
    
    
    
    
    
    





