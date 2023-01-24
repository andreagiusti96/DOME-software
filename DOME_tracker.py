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
import DOME_graphics as DOMEgraphics


def agentMatching(new_positions : np.array, positions : np.array, inactivity : List):
    """
    Track the objects in subsequent time instants assigning IDs.
    The IDs assignment is formulated as a linear optimization problem and solved with the Hungarian method.
    New IDs can be allocated.
    
    Parameters
    ----------
    new_positions : np.array (Shape=N)
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
    number_of_objects = sum(DOMEgraphics.valid_positions(positions))
    distances = np.ndarray([len(new_positions), number_of_objects])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])
    
    # build the matrix of costs
    i=0
    for pos in new_positions:
        distances[i,:] = np.squeeze(scipy.spatial.distance.cdist([pos], positions))**2
        inactivity_cost = (np.array(inactivity)**2) * 10
        distances[i,:] += inactivity_cost
        cost_newid = np.min([DOMEgraphics.distance_from_edges(pos), 100])**2 + 50
        costs_newid[i,:] = np.ones([len(new_positions)]) * cost_newid
        i+=1
        
    costs = np.concatenate((distances, costs_newid), axis=1)
    
    # Hungarian optimization algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    cost = costs[row_ind, col_ind].sum()
    
    # update ids
    new_ids = [i for i in col_ind]
    
    print('matching cost = ' + str(round(cost)) + '\t avg = ' + str(round(cost/(len(new_ids)+0.001))))
    
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
    
    print("avg speed = " + str(round(np.mean(speeds))) + "\tmax = " + str(round(max(speeds))) + "\tid =" + str(np.argmax(speeds)))
    
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
    
    estimated_pos = old_pos + velocity
    
    non_valid_pos_idx = ~ DOMEgraphics.valid_positions(estimated_pos)
    estimated_pos[non_valid_pos_idx] = estimated_pos[non_valid_pos_idx]  - velocity[non_valid_pos_idx] 
    
    return estimated_pos

def extract_data_from_images(fileLocation, area_r : List, compactness_r : List):
    
    print("Building the background model...")
    background = DOMEgraphics.build_background(fileLocation, 25)

    files = glob.glob(fileLocation +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    frames_number = len(files)
    number_of_objects=0
    n_detected_objects=0

    contours=[];
    positions= - np.ones([frames_number, 0, 2], dtype=int );
    inactivity=[]; 
    
    print("Performing detection and tracking...")
    #for counter in range(len(files)):
    for counter in range(20):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = DOMEgraphics.get_time_from_title(filename)
        print('t = ' + str(time))
        
        # collect contours and positions from new image
        new_contours = DOMEgraphics.get_contours(img, area_r=area_r, compactness_r=compactness_r, background_model=background, expected_obj_number=n_detected_objects)
        new_positions = DOMEgraphics.get_positions(new_contours)
        n_detected_objects=len(new_positions)
        
        # on first iteration assign new susequent ids to all agents
        if counter == 0: 
            new_ids = list(range(0, n_detected_objects))
        
        # on following iterations perform tracking
        else:
            est_positions=positions[counter]                  # select positions at previous time instant
            est_positions=est_positions[DOMEgraphics.valid_positions(est_positions)] # select valid positions
            new_ids = agentMatching(new_positions, est_positions, inactivity[counter])
        
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
                empty_row= - np.ones([frames_number, 1, 2], dtype=int)
                positions = np.concatenate([positions,  empty_row], axis=1)
                positions[counter, number_of_objects] = new_positions[new_ids.index(new_id)]
                contours.append(new_contours[new_ids.index(new_id)])
                inactivity.append(0)
                number_of_objects += 1
        
        # for lost objects estimate position and increase inactivity
        for lost_id in lost_obj_ids:
            inactivity[lost_id] += 1
        
        # estimate velocities and future positions
        up_to_now_positions=positions[0:counter+1]             # select positions up to current time instant
        velocities = estimate_velocities(up_to_now_positions)
        if counter < frames_number-1:
            positions[counter+1] = estimate_positions(positions[counter], velocities)
        
        # check data integrity
        assert all(DOMEgraphics.valid_positions(positions[counter]))
                
        # print image with ids and contours
        for i in range(number_of_objects):
            if inactivity[i] <=5:
                (Cx,Cy) = positions[counter][i]
                cv2.putText(img, str(i), (Cx+20,Cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,5)
                
                if inactivity[i] >0:
                    cv2.putText(img, str(inactivity[i]), (Cx+20,Cy+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) ,5)
                    cv2.drawContours(img, contours, i, (255,0,0), 4)
                    for t in range(inactivity[i]):
                        cv2.circle(img, positions[counter-t][i], 5, (255,0,0), 4)
                else :
                    cv2.drawContours(img, contours, i, (0,255,0), 4)
        DOMEgraphics.draw_image(img, 'time='+str(time) )
        
        # print info
        print('total number of objects = '+ str( number_of_objects) )
        print('detected objects = '+ str( n_detected_objects) )
        print('new ids = ' + str(newly_allocated_ids) + '\t tot = '+ str( len(newly_allocated_ids)) )
        print('total lost ids = '+ str( len(lost_obj_ids)) + '\n')
        
    return positions, inactivity


