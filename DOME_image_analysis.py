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

def draw_image(img : np.array, title : str =""):
    plt.figure(1,figsize=(20,20),dpi=72)
    plt.title(title); 
    
    if len(img.shape)==2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    plt.show()

def histogram(img : np.array):
    plt.title("Histogram");
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    
def process_img(img : np.array, color : str = "", blur  : int = 0, gain  : float = 1., contrast : bool =False, equalize : bool =False, plot : bool =False):
    
    if color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == "blue" or color == "b":
        b, g, r = cv2.split(img)
        img = b
    elif color == "green" or color == "g":
        b, g, r = cv2.split(img)
        img = g
    elif color == "red" or color == "r":
        b, g, r = cv2.split(img)
        img = r
    
    if plot: draw_image(img, "img - " + color)
    
    if gain != 1.0:
        if gain < 0: 
            min_val=img[640:1280, 360:720].min()
            #img = cv2.convertScaleAbs(img, beta=-min_val)
            img = cv2.max(float(min_val), img)
            img = img-min_val
            max_val=img[640:1280, 360:720].max()
            gain=255.0/(max_val+0.1)
            
        img = cv2.convertScaleAbs(img, alpha= gain)
        if plot: draw_image(img, "scaled")
        
    if equalize:
        img = cv2.equalizeHist(img)
        if plot: draw_image(img, "equalized")
        
    if contrast:
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16,16))
        img=clahe.apply(img)
        if plot: draw_image(img, "contrasted")
        
    if blur:
        img=cv2.medianBlur(img, blur)
        if plot: draw_image(img, "blured")
        
    return img

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
    paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))

    images = np.ndarray([images_number, 1080, 1920], dtype=np.uint8)
    indices = np.linspace(0, len(paths)-1, num=images_number, dtype=int)
    #indices = np.linspace(0, images_number-1, num=images_number, dtype=int)

    selected_paths = [paths[i] for i in indices]

    counter=0
    for filename in selected_paths:
        img = cv2.imread(filename)
        # Convert the frame to grayscale
        elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)
        #draw_image(elaborated_img, "img "+str(counter))
        images[counter] = elaborated_img
        counter+=1
    
    # compute background
    background = np.median(images, axis=0)
    #background = np.mean(images, axis=0)
    background = background.round().astype(np.uint8)
    background = np.squeeze(background)
    
    draw_image(background, "background from color "+DEFAULT_COLOR)
    
    return background
   
    
def get_contours(img : np.array, area_r : List, compactness_r : List, background_model=None, expected_obj_number : int =0): 
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
    #draw_image(img, "img")

    elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)

    # Subtract the background from the frame
    if type(background_model)==type(None): background_model= np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    foreground = cv2.min(cv2.absdiff(elaborated_img, background_model), elaborated_img)
    #draw_image(foreground, "foreground")
    
    foreground = process_img(foreground, blur=DEFAULT_BLUR)
    # draw_image(foreground, "elaborated foreground")
    
    threshold = 122
    a_r=area_r.copy()
    c_r=compactness_r.copy()
    
    first_time=True
    margin_factor = 0.25
    adjustment_factor = 0.05
    while first_time or (len(contoursFiltered) < expected_obj_number * (1-margin_factor) and threshold > 55) or (len(contoursFiltered) > expected_obj_number * (1+margin_factor) and threshold<200):
        first_time=False
        
        # Apply thresholding to the foreground image to create a binary mask
        ret, mask = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)
        #draw_image(mask, "mask")
    
        # Find contours of objects
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_img=img.copy()
        cv2.drawContours(contours_img, contours, -1, (0,255,0), 3)
    
        contoursFiltered=[]
        for i in range(len(contours)):
            contour=contours[i]
            area = cv2.contourArea(contour)
            
            # print contours info
            (Cx,Cy) = np.squeeze(get_positions(contours[i:i+1]))
            cv2.putText(contours_img, "A="+str(round(area)), (Cx+20,Cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,4)
            
            if a_r[0] <= area <= a_r[1]:
                perimeter = cv2.arcLength(contour,True)
                compactness=(4*np.pi*area)/(perimeter**2) #Polsby–Popper test
                cv2.putText(contours_img, "C="+str(round(compactness,2)), (Cx+20,Cy+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,4)
                
                if c_r[0] <= compactness <= c_r[1]:
                    contoursFiltered.append(contour)
        
        
        if expected_obj_number==0: expected_obj_number = len(contoursFiltered)
        
        #draw_image(contours_img, "contours with thresh=" +str(threshold))
        contoursFiltered_img=cv2.cvtColor(foreground,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursFiltered_img, contoursFiltered, -1, (0,255,0), 3)
        draw_image(contoursFiltered_img, "contoursFiltered with thresh=" +str(threshold))
        
        print("thresh="+ str(round(threshold)) +"\t area_r="+ str(np.around(a_r)) +"\t compactness_r="+ str(np.around(c_r,2)) +"\t objects=" + str(len(contoursFiltered))+"\t exp objects=" + str(expected_obj_number))
        if len(contoursFiltered) < expected_obj_number * (1-margin_factor) :
            threshold=threshold * (1-adjustment_factor)
            a_r[0]=a_r[0] * (1-adjustment_factor)
            a_r[1]=a_r[1] * (1+adjustment_factor)            
            c_r[0]=c_r[0] * (1-adjustment_factor)
            c_r[1]=c_r[1] * (1+adjustment_factor)
        elif len(contoursFiltered) > expected_obj_number * (1+margin_factor) :
            threshold=threshold * (1+adjustment_factor)
            a_r[0]=a_r[0] * (1+adjustment_factor)
            a_r[1]=a_r[1] * (1-adjustment_factor)            
            c_r[0]=c_r[0] * (1+adjustment_factor)
            c_r[1]=c_r[1] * (1-adjustment_factor)
        
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
        cost_newid = np.min([distance_from_edges(pos), 100])**2 + 50
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
    
    background = build_background(fileLocation, 25)

    files = glob.glob(fileLocation +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    frames_number = len(files)
    number_of_objects=0
    n_detected_objects=0

    contours=[];
    positions= - np.ones([frames_number, 0, 2], dtype=int );
    inactivity=[]; 
    
    for counter in range(len(files)):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = get_time_from_title(filename)
        print('\nt = ' + str(time))
        
        # collect contours and positions from new image
        new_contours = get_contours(img, area_r=AREA_RANGE, compactness_r=COMPAC_RANGE, background_model=background, expected_obj_number=n_detected_objects)
        new_positions = get_positions(new_contours)
        n_detected_objects=len(new_positions)
        
        # on first iteration assign new susequent ids to all agents
        if counter == 0: 
            new_ids = list(range(0, n_detected_objects))
        
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
        draw_image(img, 'time='+str(time) )
        
        # print info
        print('total number of objects = '+ str( number_of_objects) )
        print('detected objects = '+ str( n_detected_objects) )
        print('new ids = ' + str(newly_allocated_ids) + '\t tot = '+ str( len(newly_allocated_ids)) )
        print('total lost ids = '+ str( len(lost_obj_ids)) )
        
def write_to_file(velocity_list):
    df = pandas.DataFrame(velocity_list) 
    df.to_csv('velocity_list.csv') 


# MAIN

filePath = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments/2022_12_19_Euglena_3'

DEFAULT_COLOR = "green"
DEFAULT_BLUR = 9
AUTO_SCALING = -1
AREA_RANGE = [100, 600]
COMPAC_RANGE = [0.5, 0.9]

imageImport(filePath)



