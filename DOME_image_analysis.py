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
import matplotlib.pyplot as plt
import pandas 
from typing import List
import re


def build_background(fileLocation : str, images_number : int):
    """
    Extract the background from a set of images excluding moving objects.
    Background is computed as the median pixel-wise of the images.
    Camera has to be static.
    
    Parameters
    ----------
    fileLocation : str
        Path of the folder containing the images.
    images_number : int
        Number of images to use to build the background.

    Returns
    -------
    background : np.array
        Gray scale image of the background.

    """
    paths=glob.glob(fileLocation +  '/*.jpeg')
    
    images = np.ndarray([images_number, 1080, 1920], dtype=np.uint8)
    indices = np.linspace(0, len(paths)-1, num=images_number, dtype=int)

    selected_paths = [paths[i] for i in indices]

    counter=0
    for filename in selected_paths:
        img = cv2.imread(filename)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images[counter] = gray
        counter+=1
    
    # compute median
    background = np.median(images, axis=0)
    background = background.round().astype(np.uint8)
    background = np.squeeze(background)
    plt.title('background'); plt.imshow(background, cmap='gray', vmin=0, vmax=255); plt.show()
    return background

def get_contours(img : np.array, min_area : float, min_compactness : float, background_model=None):
    """
    Thresholding based object detection.

    Parameters
    ----------
    img : np.array
        Image to analyse.
    min_area : float
        Minimum area of objects in pixels.
    min_compactness : float
        Minimum compactness of objects [0, 1].
    background_model : np.array, optional
        Image of the background to perform background subtraction. 
        The default is None.

    Returns
    -------
    contoursFiltered : List
        Contours of the detected objects.

    """
    contoursFiltered=[]
    #plt.title('img'); plt.imshow(img); plt.show()
    
    # Convert the frame to grayscale and apply histogram equalization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #equalized = cv2.equalizeHist(gray)
    #plt.title('gray');  plt.imshow(gray, cmap='gray', vmin=0, vmax=255); plt.show()
    
    # Subtract the background from the frame
    if type(background_model)==type(None): background_model= np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    foreground = cv2.absdiff(gray, background_model)
    #plt.title('foreground'); plt.imshow(foreground, cmap='gray', vmin=0, vmax=255); plt.show()

    # Apply thresholding to the foreground image to create a binary mask
    ret, mask = cv2.threshold(foreground, 100, 255, cv2.THRESH_BINARY)
    #plt.title('mask'); plt.imshow(mask, cmap='gray', vmin=0, vmax=255); plt.show()
    
    # Find contours of objects
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_img=img.copy()
    cv2.drawContours(contours_img, contours, -1, (0,255,0), 3)
    #plt.title('contours'); plt.imshow(contours_img); plt.show()
    
    contoursFiltered=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour,True)
        if area > min_area:
            compactness=(4*np.pi*area)/(perimeter**2) #Polsby–Popper test
            if compactness > min_compactness:
                contoursFiltered.append(contour)
    
    #contoursFiltered=np.array(contoursFiltered)
    contoursFiltered_img=img.copy()
    cv2.drawContours(contoursFiltered_img, contoursFiltered, -1, (0,255,0), 3)
    #plt.title('contoursFiltered'); plt.imshow(contoursFiltered_img); plt.show()
    
    return contoursFiltered

def get_positions(contours):
    """
    Get the centers of contours resulting from image analysis

    Parameters
    ----------
    contours : List
        Contours of the detected objects. (Shape=Nx2)

    Returns
    -------
    positions : List
        Position of the center of each object. (Shape=Nx2)

    """
    positions = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        positions.append([x+int(w/2),y+int(h/2)])
    
    return positions

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
    number_of_objects = sum(valid_positions(positions))
    distances = np.ndarray([len(new_positions), number_of_objects])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])
    
    # build the matrix of costs
    i=0
    for pos in new_positions:
        distances[i,:] = np.squeeze(scipy.spatial.distance.cdist([pos], positions))**2
        inactivity_cost = (np.array(inactivity)**2) * 10
        distances[i,:] += inactivity_cost
        cost_newid = np.min([distance_from_edges(pos), 100])**2 + 25
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

def distance_from_edges(pos : List):
    """
    Get the distance from the closest edge of the picture frame.

    Parameters
    ----------
    pos : List
        Position. Shape=(2)

    Returns
    -------
    dis : float or int depending on the input
        Distances from the closest edge.

    """
    assert(len(pos)==2)
    
    dis= np.min([pos[0], pos[1], 1920-pos[0], 1080-pos[1]])
    
    assert(dis>=0)
    return dis

def get_time_from_title(filename: str):
    """
    Extract time from a string.

    Parameters
    ----------
    filename : str
        String to extract the time from.

    Returns
    -------
    time : float
        Time sxtracted from the string.

    """
    file_name = filename.split("fig_")[-1]
    file_name = file_name.split(".jpeg")[0]
    time = float(file_name)
    return time

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
        valid_pos_idx = valid_positions(positions[-2])
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
    
    non_valid_pos_idx = ~ valid_positions(estimated_pos)
    estimated_pos[non_valid_pos_idx] = estimated_pos[non_valid_pos_idx]  - velocity[non_valid_pos_idx] 
    
    return estimated_pos

def valid_positions(positions : np.array):
    """
    Get the indices of valid positions, i.e. positions in the range [0, 1920][0, 1080]

    Parameters
    ----------
    positions : np.array
        Array of positions. Shape=(Nx2)

    Returns
    -------
    validity : np.array
        Array of bool telling whether the corresponding position is valid.

    """
    validity0=(positions[:,0] >= 0) & (positions[:,0] <= 1920)
    validity1=(positions[:,1] >= 0) & (positions[:,1] <= 1080)
    validity = validity0 * validity1
    
    assert len(validity) == positions.shape[0]
    return validity

def imageImport(fileLocation):
    
    background = build_background(fileLocation, 20)

    files = glob.glob(fileLocation +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    frames_number = len(files)
    number_of_objects=0
    
    contours=[];
    positions= - np.ones([frames_number, 0, 2], dtype=int );
    inactivity=[]; 
    
    counter = 0 
    for filename in files:
        # declare vars
        img = cv2.imread(filename)
        time = get_time_from_title(filename)
        print('\nt = ' + str(time))
            
        # collect contours and positions from new image
        new_contours = get_contours(img, min_area=150, min_compactness=0.25, background_model=background)
        new_positions = get_positions(new_contours)
        
        # on first iteration assign new susequent ids to all agents
        if counter == 0: 
            new_ids = list(range(0, len(new_positions)))
        
        # on following iterations perform tracking
        else:
            est_positions=positions[counter]                  # select positions at previous time instant
            est_positions=est_positions[valid_positions(est_positions)] # select valid positions
            new_ids = agentMatching(new_positions, est_positions, inactivity)
        
        # discern new and lost objects
        newly_allocated_ids = [i for i in new_ids if i not in range(number_of_objects)]
        lost_obj_ids = [i for i in range(number_of_objects) if i not in new_ids]
        
        # update data
        for new_id in new_ids:
            # for already detected objects update data
            if new_id < number_of_objects:
                positions[counter, new_id] = new_positions[new_ids.index(new_id)]
                contours[new_id] = new_contours[new_ids.index(new_id)]
                inactivity[new_id] = 0
                
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
        assert all(valid_positions(positions[counter]))
                
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
        plt.figure(1,figsize=(20,20),dpi=72)
        plt.title('time='+str(time)); plt.imshow(img); plt.show()
        
        # print info
        print('number of objects = '+ str( number_of_objects) )
        print('new ids = ' + str(newly_allocated_ids) + '\t tot = '+ str( len(newly_allocated_ids)) )
        print('lost ids = '+ str( len(lost_obj_ids)) )
        
        counter += 1

def write_to_file(velocity_list):
    df = pandas.DataFrame(velocity_list) 
    df.to_csv('velocity_list.csv') 


# MAIN

filePath = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments/2022_12_19_Euglena_3'
imageImport(filePath)



