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
import glob
import matplotlib.pyplot as plt
import random
import math 
import pandas 


#reads in new images and performs image analysis to find which contours relate to agents
def image_analysis(img, img_init):
    frame_new = img
    contoursFiltered=[]
    b,g,r = cv2.split(frame_new)
    ret, thresh2 = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour,True)
        if area > 100:
            compactness=(4*np.pi*area)/(perimeter**2)
            if compactness > 0.5:
                (x,y,w,h) = cv2.boundingRect(contour)
                contoursFiltered.append([x+int(w/2),y+int(h/2),w,h,1,0])
    contoursFiltered=np.array(contoursFiltered)
    return contoursFiltered


#find the best match between agents in frames to assign continous ID
def agentMatching(past_contours, img_contours):
    #initalise agent matching array
    matched_agents = np.copy(past_contours)
    #set propagating agents to 1
    #caculate length of deactivity, becomes increasingly negative the more frames are dropped
    #note that if the countour is matched in the subsequent section, is will reset to zero
    for agent in matched_agents:
        agent[5]=agent[5]+(agent[4]-1)
    #set current activity to deactice for all agents
    matched_agents[0:len(matched_agents), 4]=0
    for agent in img_contours:
        position_difference_match = 10000 #defines how close agents can be to be considered the same agent
        agent_id = 0
        for past_agent in past_contours:
            #excludes long term deactivated agents
            if past_agent[5] > -3:
                position_difference=(abs(agent[0]-past_agent[0]), abs(agent[1]-past_agent[1]))
                position_difference=sum(position_difference)
                if position_difference < position_difference_match:
                    contour_match = agent
                    past_contours_match = past_agent
                    matched_agent_id = agent_id
                    position_difference_match=position_difference  
            else:
                pass
            agent_id+=1
        #print("MATCH CON\n", past_contours_match, "CONFIDENCE", position_difference_match, "ID", matched_agent_id)
        #if the agent falls outside of a given confidence inteval, assume new agents and append to the end of the array 
        if position_difference_match > 30:
            contour_match = np.array([contour_match])
            matched_agents= np.append(matched_agents, contour_match, axis = 0)
        else:
            matched_agents[matched_agent_id]=contour_match
    #carry over propogation number
    return matched_agents


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
    counter = 0 
    velocity_list=[]
    
    fileNames = sorted(glob.glob(fileLocation +  '/*.jpeg'))
    
    for filename in fileNames:
        img = cv2.imread(filename)
        time = get_time_from_title(filename)
        blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # for first instance of loop, initialise time zero
        if counter == 0: 
            img_init = img
            time_zero = time
        
        # for all other loop instances, match agents and calculate velocity or displacement
        else:
            time_difference = (time-time_zero)*10
            img_contours = image_analysis(img, img_init)
            if counter == 1: 
                agents_predictive = img_contours
                for i in range(len(agents_predictive)):
                    velocity_list.append([])
            else:
                img_contours = agentMatching(past_contours, img_contours)
                velocity_list = displacement_calculation(velocity_list, past_contours, img_contours, time_difference, counter)
            
            agent_id = 0
            for c in img_contours:
                active_status = c[4]
                if active_status == 1:
                    (Cx,Cy) = (c[0],c[1]+int(c[2]/2))
                    radius  = 10
                    cv2.putText(img, str(agent_id), (Cx,Cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,2)
                agent_id += 1 
            cv2.imshow("Image", img)
            cv2.waitKey(33)
            past_contours = img_contours
        
        counter += 1
        #k=cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if counter == 20:
            break
    for item in velocity_list:
        print(len(item))
    return velocity_list

def write_to_file(velocity_list):
    df = pandas.DataFrame(velocity_list) 
    df.to_csv('velocity_list.csv') 

filePath = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments/2022_12_19_Euglena_3'
velocity_list = imageImport(filePath)
write_to_file(velocity_list)


