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
import re
from typing import List



def draw_image(img : np.array = np.zeros([1080, 1920]), title : str =""):
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
   
def get_img_at_time(fileLocation : str, image_time : float):
    paths=glob.glob(fileLocation +  '/*.jpeg')
    paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))

    for filename in paths:
        if get_time_from_title(filename) == image_time:
            img = cv2.imread(filename)
            
    return img
    
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
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        #draw_image(contoursFiltered_img, "contoursFiltered with thresh=" +str(threshold))
        
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

def std_color_for_index(index : int):
    index = index%len(STD_COLORS)
    color = STD_COLORS[index]
    return color

# CONSTANTS
DEFAULT_COLOR = "green"
DEFAULT_BLUR = 9
AUTO_SCALING = -1

STD_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


