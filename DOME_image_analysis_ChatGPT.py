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
import glob

# Load the reference template image
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

# Set the detection threshold
threshold = 0.9

# Load the images
image_paths = glob.glob("*.jpg")
image_paths.sort()

# Create a background model by averaging a set of frames with no moving objects
frame_count = 0
background_model = None
for image_path in image_paths:
    # Read the next frame
    frame = cv2.imread(image_path)
    frame_count += 1
    
    # Convert the frame to grayscale and apply histogram equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Update the background model
    if background_model is None:
        background_model = equalized
    else:
        background_model = (background_model * (frame_count - 1) + equalized) / frame_count

# Process the remaining frames of the images
for image_path in image_paths:
    # Read the next frame
    frame = cv2.imread(image_path)
    
    # Convert the frame to grayscale and apply histogram equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Subtract the background from the frame
    foreground = cv2.absdiff(equalized, background_model)
    
    # Apply thresholding to the foreground image to create a binary mask
    ret, mask = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)
    
    # Perform template matching to find instances of the object
    res = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
    loc = cv2.minMaxLoc(res)[-2]
    
    # Check if the detection is above the threshold
    if res[loc[1]][loc[0]] > threshold:
        # Draw a bounding box around the detected object
        top_left = (loc[0], loc[1])
        bottom_right = (loc[0] + template.shape[1], loc[1] + template.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

# Destroy the window
cv2.destroyAllWindows()

    
    
# MAIN

filePath = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments/2022_12_19_Euglena_3'
images, activation_times = importImages(filePath)


