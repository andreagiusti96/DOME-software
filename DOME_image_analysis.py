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
import random
import math 
import pandas 
from typing import List


def build_background(fileLocation : str, images_number : int):
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

#reads in new images and performs image analysis to find which contours relate to agents
def get_contours(img, min_area : float, min_compactness : float, background_model=None):
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
    positions = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        positions.append([x+int(w/2),y+int(h/2)])
    
    return positions

# #find the best match between agents in frames to assign continous ID
# def agentMatching(new_positions, old_positions, old_ids):
#     new_ids=[]
#    
#     for agent in new_positions:
#         distances = np.squeeze(scipy.spatial.distance.cdist([agent], old_positions))
#         for new_id in new_ids: distances[old_ids.index(new_id)]=np.inf 
#         new_ids.append(old_ids[np.argmin(distances)])
#
#     return new_ids

# Tracking (Hungarian method)
def agentMatching(new_positions, old_positions, old_ids):
    new_ids=[]
    distances = np.ndarray([len(new_positions), len(old_positions)])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])
    
    newly_allocable_ids_range = range(np.max(old_ids)+1, np.max(old_ids)+costs_newid.shape[1]+1)
    available_ids = old_ids + list(newly_allocable_ids_range)
    
    # build the matrix of costs
    i=0
    for pos in new_positions:
        distances[i,:] = np.squeeze(scipy.spatial.distance.cdist([pos], old_positions))
        cost_newid = np.min([distance_from_edges(pos), 100]) + 50
        costs_newid[i,:] = np.ones([len(new_positions)]) * cost_newid
        i+=1
        
    costs = np.concatenate((distances, costs_newid), axis=1)
    
    # Hungarian optimization algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    cost = costs[row_ind, col_ind].sum()
    
    # update ids
    new_ids = [available_ids[i] for i in col_ind]
    
    print('avg cost = ' + str(cost/len(new_ids)))
    newly_allocated_ids = [i for i in new_ids if i not in old_ids]
    lost_ids = [i for i in old_ids if i not in new_ids]
    print('new ids = ' + str(newly_allocated_ids) + '\t tot = '+ str( len(newly_allocated_ids)) )
    print('lost ids = ' + str(lost_ids) + '\t tot = '+ str( len(lost_ids)) )
    
    return new_ids

def distance_from_edges(pos : List):
    assert(len(pos)==2)
    
    dis= np.min([pos[0], pos[1], 1920-pos[0], 1080-pos[1]])
    
    assert(dis>=0)
    return dis

def displacement_calculation(velocity_list, past_contours, img_contours, time_difference, counter):
    new_agents = len(img_contours)- len(past_contours)
    if new_agents > 0:
        for i in range(new_agents):
            velocity_list.append([])
    try:
        for i in range(len(img_contours)):
            if img_contours[i][5]>-3:
                (x,y) = img_contours[i][0], img_contours[i][1]
                (x_past, y_past) = past_contours[i][0], past_contours[i][1]
                #velocity_direction=[img_contours[i][0]-past_contours[i][0], img_contours[i][1]-past_contours[i][1]]
                displacement=(x-x_past , y- y_past)
                total_displacement = math.sqrt(displacement[0]**2+displacement[1]**2)
                #velocity = total_displacement/time_difference
                velocity_list[i].append((total_displacement, time_difference, counter))
    except:
        print("Agent", i, "not found")
    return velocity_list

def get_time_from_title(filename: str):
    file_name = filename.split("fig_")[-1]
    file_name = file_name.split(".jpeg")[0]
    time = float(file_name)
    return time

def imageImport(fileLocation):
    velocity_list=[]
    old_contours=[];
    old_positions=[];
    old_ids=[];
    
    background = build_background(fileLocation, 20)

    files=sorted(glob.glob(fileLocation +  '/*.jpeg'))
    counter = 0 
    for filename in files:
        # declare vars
        img = cv2.imread(filename)
        time = get_time_from_title(filename)
        print('t = ' + str(time))
            
        # collect contours and positions from new image
        new_contours = get_contours(img, min_area=200, min_compactness=0.25, background_model=background)
        new_positions = get_positions(new_contours)
        
        # on first iteration assign new susequent ids to all agents
        if counter == 0: 
            new_ids = list(range(0, len(new_positions)))
            
        
        # on following iterations perform tracking
        else:
            new_ids = agentMatching(new_positions, old_positions, old_ids)
            #velocity_list = displacement_calculation(velocity_list, old_contours, new_contours, time, counter)
        
        # update old data
        old_contours = new_contours.copy()        
        old_positions = new_positions.copy()
        old_ids = new_ids.copy()
        
        # print image with ids and contours
        for i in range(len(new_positions)):
            (Cx,Cy) = (new_positions[i][0], new_positions[i][1]+40)
            cv2.putText(img, str(new_ids[i]), (Cx,Cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,5)
        cv2.drawContours(img, new_contours, -1, (0,255,0), 4)
        plt.title('time='+str(time)); plt.imshow(img); plt.show()
        
        counter += 1
    
    return velocity_list

def write_to_file(velocity_list):
    df = pandas.DataFrame(velocity_list) 
    df.to_csv('velocity_list.csv') 


# MAIN

filePath = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments/2022_12_19_Euglena_3'
velocity_list = imageImport(filePath)



