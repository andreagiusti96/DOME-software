#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code reads data collected during a DOME experiment and performs objects detection and tracking.
When you launch the tracking of an experiment the algorithm:
1) Builds the background model representing static elements in the camera frame
2) For each frame perfoms thresholding based objects detection and tracking.
3) Generates tracking images and video.
4) Saves the resulting tracking data.

Run this script, then follow instructions in the console.

Author:     Andrea Giusti
Created:    2023
"""

import cv2
import numpy as np
import scipy
import glob
import re
import os
import matplotlib.pyplot as plt
import random
from datetime import datetime
from typing import List


import DOME_graphics as DOMEgraphics
import DOME_experiment_manager as DOMEexp

NEW_ID_COST_MIN = 1
NEW_ID_COST_DIST_CAP = 3
DISTANCE_COST_FACTORS = [0, 1]
INACTIVITY_COST_FACTORS = [0, 1]


def matchingCost(distance : np.array, inactivity : int):
    """
    Compute the matching cost(s) of one object. Given the distances from the expected positions and its inactivity. 

    Parameters
    ----------
    distance : np.array
        Vector of distances from possible expected positions.
    inactivity : int
        Currente inactivity indicator of the object..

    Returns
    -------
    cost : np.array
        Matching costs.
    """
    cost = (distance * DISTANCE_COST_FACTORS[0] + distance ** 2 * DISTANCE_COST_FACTORS[1]) / (
                inactivity ** 2 * 0.25 + 1)
    cost += inactivity * INACTIVITY_COST_FACTORS[0] + inactivity ** 2 * INACTIVITY_COST_FACTORS[1]
    return cost


def plotCosts():
    """
    Plot the matching costs as a function of the distance and for different inactivity values.

    Returns
    -------
    None.
    """
    fig = plt.figure(1, figsize=(19.20, 10.80), dpi=100)
    fig.subplots_adjust(top=1.0 - 0.05, bottom=0.05, right=1.0 - 0.05, left=0.05, hspace=0, wspace=0)
    plt.title('Matching cost')

    maxdist = 4
    distances = np.linspace(0, maxdist)
    inactivity = np.array([0, 1, 2, 3, 4, 5])

    matching_cost = np.zeros([len(distances), len(inactivity)])

    for i in range(len(inactivity)):
        matching_cost[:, i] = matchingCost(distances, inactivity[i])

    new_id_cost_max = (NEW_ID_COST_DIST_CAP ** 2) + NEW_ID_COST_MIN

    plt.plot(distances, matching_cost)
    plt.plot([0, maxdist], NEW_ID_COST_MIN * np.array([1, 1]))
    plt.plot([0, maxdist], new_id_cost_max * np.array([1, 1]))
    plt.legend(inactivity)
    plt.gca().set_ylim([0, new_id_cost_max * 1.5])
    plt.gca().set_xlim([0, maxdist])
    plt.xlabel('distance/(TYPICAL_VEL*deltaT)')


def agentMatching(new_positions: np.array, positions: np.array, inactivity: List, deltaT : float, verbose:bool=False):
    """
    Track the objects in subsequent time instants assigning existent IDs or allocating new ones.
    The IDs assignment is formulated as a linear optimization problem and solved with the Hungarian method.
    New IDs can be allocated.

    Parameters
    ----------
    new_positions : np.array of float Shape=(Nx2)
        Positions of detected objects.
    positions : np.array of float
        Positions of previously detected objects. Shape=(Mx2)
    inactivity : List
        Inactivity counters of the objects. Shape=(M)
    deltaT : float
        Time interval from last sampling [seconds].
    verbose : bool = False
        If True print matching costs.
    
    Returns
    -------
    new_ids : List
        IDs assigned to the detected positions. Shape=(N)
    cost : float
        Total matching cost.

    """

    # allocate vars
    new_positions = np.array(new_positions)
    number_of_objects = sum(valid_positions(positions))
    costs_matching = np.ndarray([len(new_positions), number_of_objects])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])

    assert len(new_positions.shape) == 2
    assert len(positions.shape) == 2
    assert new_positions.shape[1] == 2
    assert positions.shape[1] == 2
    
    TYPICAL_VEL = PARAMETERS["TYPICAL_VEL"]

    # compute distances between all possible pairs (estimated positions, detected positions)
    distances = np.squeeze(scipy.spatial.distance.cdist(new_positions, positions))
    distances = distances / (TYPICAL_VEL*deltaT)

    # build the matrix of costs
    # compute matching costs
    for i in range(positions.shape[0]):
        costs_matching[:, i] = matchingCost(distances[:, i], inactivity[i])
    
    # compute costs for the allocation of new IDs
    for i in range(new_positions.shape[0]):
        cost_newid = np.min(
            [distance_from_edges(new_positions[i]) / (TYPICAL_VEL*deltaT), NEW_ID_COST_DIST_CAP]) ** 2 + NEW_ID_COST_MIN
        costs_newid[i, :] = np.ones([len(new_positions)]) * cost_newid

    costs = np.concatenate((costs_matching, costs_newid), axis=1)

    # use Hungarian optimization algorithm to minimize the total matching cost
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    cost = costs[row_ind, col_ind].sum()*TYPICAL_VEL

    # update list of IDs
    new_ids = [i for i in col_ind]
    avg_cost = cost / (len(new_ids) + 0.001)

    if verbose:
        print('matching cost = ' + str(round(cost, 2)) + '\t avg = ' + str(round(avg_cost, 2)))

    return new_ids, cost


def estimate_velocities(positions: np.array, deltaT : float):
    """
    Given the past positions of the objects estimates their velocities.

    Parameters
    ----------
    positions : np.array of float
        Past positions of the objects. Shape=(MxNx2)
        If M<2 all velocities are [0, 0].
        Non valid position are discarded.
    deltaT : float
        Time interval from last sampling [seconds].
    
    Returns
    -------
    velocities : np.array
        Velocities of the objects. Shape=(Nx2)

    """
    assert len(positions.shape) == 3
    assert positions.shape[2] == 2
    assert deltaT > 0

    velocities = np.zeros(positions.shape[1:3])

    if positions.shape[0] >= 2:
        valid_pos_idx = valid_positions(positions[-2])
        velocities[valid_pos_idx] = positions[-1, valid_pos_idx] - positions[-2, valid_pos_idx]
        velocities = velocities/ deltaT

    # speeds = np.linalg.norm(velocities, axis=1)
    # print("avg speed = " + str(round(np.mean(speeds),1)) + "\tmax = " + str(round(max(speeds),1)) + "\tid =" + str(np.argmax(speeds)))

    assert velocities.shape[1] == 2
    return velocities


def estimate_positions(old_pos: np.array, velocity: np.array, deltaT:float):
    """
    Given the current positions and velocities returns the future estimated positions of objects.
    The INERTIA coefficent multiplies the velocity. 
    Positions are validated to be in the range [0, 1920][0, 1080]

    Parameters
    ----------
    old_pos : np.array
        Last positions of the objects. Shape=(Nx2)
    velocity : np.array
        Velocities of the objects. Shape=(Nx2)
    deltaT : float
        Time interval from last sampling [seconds].

    Returns
    -------
    estimated_pos : np.array
        Estimated positions of the objects at the next time instant. Shape=(Nx2).

    """
    assert len(old_pos.shape) == 2
    assert len(velocity.shape) == 2
    assert old_pos.shape[1] == 2
    assert deltaT > 0

    INERTIA = PARAMETERS["INERTIA"]

    estimated_pos = old_pos + velocity * INERTIA * deltaT 

    non_valid_pos_idx = ~ valid_positions(estimated_pos)
    estimated_pos[non_valid_pos_idx] = estimated_pos[non_valid_pos_idx] - velocity[non_valid_pos_idx] * INERTIA * deltaT 

    return estimated_pos


def interpolate_positions(positions: np.array, original_times : List = [], new_times : List = []):
    """
    Interpolate positions at given time instants, and eventually replace internal nan values.
    If original_times and new_times are empty datapoints are assumed to be equispaced.
    Otherwise, values at new_times are computed using the given values at original_times.
    
    Parameters
    ----------
    positions : np.array of float
        Array of 2D positions, possibly containing some nans. Shape=(MxNx2).
    original_times : List = []
        List of original sampling times. Optional.
    new_times : List = []
        List of new (equally spaced) time instants at which positions must be computed. Optional. Shape=(M_new)

    Returns
    -------
    interpolated_pos_new : np.array
        Array of 2D interpoleted positions.
        Internal nans have been replaced by linear interpolation.
        Shape=(M_newxNx2) if new_times was given, Shape=(MxNx2) otherwise.

    Example
    -------
    Simplyfied case with 1D positions, original_times = [] and new_times = []:
    Input:
        positions = [nan, nan, 1, 2, nan, 4, nan]
        original_times = []
        new_times = []
    Output:
        interpolated_pos_new = [nan, nan, 1, 2, 3, 4, nan]
    
    """
    # allocate vars
    interpolated_pos = positions.copy()
    number_objects = positions.shape[1]
    interpolated_pos_new = np.ndarray([len(new_times), number_objects, 2]) * np.nan

    # for each object in positions
    for obj in range(positions.shape[1]):
        # ignore leading and trailing nans and select internal points
        first_index = (~np.isnan(positions[:, obj, 0])).argmax(0)
        last_index = positions.shape[0] - (~np.isnan(positions[:, obj, 0]))[::-1].argmax(0) - 1
        active_points = slice(first_index,last_index+1)
        
        # select nans within internal points
        nans = np.isnan(positions[active_points, obj, 0])
        missing_points = np.where(nans)[0] + first_index
        valid_points = np.where(~nans)[0] + first_index

        # replace internal nans
        if len(missing_points) and len(valid_points)> 0:
            trajectory_x = positions[valid_points, obj, 0]
            trajectory_y = positions[valid_points, obj, 1]
            interpolated_pos[missing_points, obj, 0] = np.interp(missing_points, valid_points, trajectory_x)
            interpolated_pos[missing_points, obj, 1] = np.interp(missing_points, valid_points, trajectory_y)
        
        # if original_times is given use it to compute values at new_times
        if len(original_times)>0:
            interpolated_pos_new[active_points, obj, 0] = np.interp(new_times[active_points], original_times[active_points], interpolated_pos[active_points,obj,0])
            interpolated_pos_new[active_points, obj, 1] = np.interp(new_times[active_points], original_times[active_points], interpolated_pos[active_points,obj,1])
        else:
            interpolated_pos_new = interpolated_pos

        # print(np.concatenate([positions[:last_index+2,obj], interpolated_pos[:last_index+2,obj]], axis=1))
    return interpolated_pos_new


def test_detection_parameters(fileLocation : str, bright_thresh : int, area_r: List, compactness_r: List):
    """
    Test the objects detection algorithm and parameters, showing step by step images.
    First the background model is built, then the object detection is tested on an image.
    
    Parameters
    ----------
    fileLocation : str
        Path to a folder containg a set of images or to a specific image in that folder.
        If the path of an image is given that image is used to test the algorithm.
        Otherwise, a random image from the folder is used.
    bright_thresh : int
        Minimum brightness used for thresholding.
    area_r : List
        Two positive values defining the range of area for obj detection [pixels].
    compactness_r : List
        Two values in [0,1] defining the range of compactness for obj detection.

    Returns
    -------
    None.

    """
    if os.path.isdir(fileLocation):
        files = glob.glob(fileLocation + '/*.jpeg')
        files = sorted(files, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))
        filename = random.choice(files)
    elif os.path.isfile(fileLocation):
        filename = fileLocation
        fileLocation = os.path.dirname(fileLocation)
    else:
        raise (Exception(f'Not a file or a directory: {fileLocation}'))
    
    # get the image to test object detection
    img = cv2.imread(filename)

    # build the model of the background using images in the folder
    background = build_background(fileLocation, 25)

    # test objects detection showing step by step images
    new_contours = get_contours(img, bright_thresh, area_r, compactness_r, background, 0, plot=True)


def process_img(img: np.array, color: str = "", blur: int = 0, gain: float = 1., 
                contrast: bool = False, equalize: bool = False, plot: bool = False):
    """
    Process an image based on the provided parameters.
    Possible eleborations include color channel extraction, blurring, gain adjustment, 
    histogram equalization, and contrast enhancement. 
    The eleborated image can be ploted at each step if requested.

    Parameters
    ----------
    img : np.array
        The input image to be processed.
    color : str = ""
        The color channel to extract from the image. Options are "gray", "blue" or "b", "green" or "g", and "red" or "r". Default is "" (no color extraction).
    blur : int = 0
        The kernel size for the median blur. If 0, no blurring is applied.
    gain : float = 1
        The grightness gain. 
        If less than 0, the brightness is automatically scaled so that the maximum pixel value is 255, and the minimum is 0. 
        Default is 1.0 (no gain adjustment).
    contrast : bool = False
        If True, applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image. 
    equalize : bool = False
        If True, equalizes the histogram of the image.
    plot : bool = False
        If True, plots the image at each processing step.

    Returns
    -------
    img : np.array
        The processed image.
    """
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

    if plot: DOMEgraphics.draw_image(img, "img - " + color)

    if gain != 1.0:
        if gain < 0:
            min_val = img[640:1280, 360:720].min()
            # img = cv2.convertScaleAbs(img, beta=-min_val)
            img = cv2.max(float(min_val), img)
            img = img - min_val
            max_val = img[640:1280, 360:720].max()
            gain = 255.0 / (max_val + 0.1)

        img = cv2.convertScaleAbs(img, alpha=gain)
        if plot: DOMEgraphics.draw_image(img, "scaled")

    if equalize:
        img = cv2.equalizeHist(img)
        if plot: DOMEgraphics.draw_image(img, "equalized")

    if contrast:
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))
        img = clahe.apply(img)
        if plot: DOMEgraphics.draw_image(img, "contrasted")

    if blur:
        img = cv2.medianBlur(img, blur)
        if plot: DOMEgraphics.draw_image(img, "blured")

    return img


def build_background(fileLocation: str, images_number : int = 10, gain: float = 1.0):
    """
    Extract the background from a set of images by excluding moving objects.
    First images are preprocessed as during the object detection.
    Then, the background is computed as the pixel-wise median of the images.
    The camera is assumed to be static.

    Parameters
    ----------
    fileLocation : str
        Path of the folder containing a set of images.
    images_number : int = 10
        Number of images to be used to build the background.
    gain : float = 1.0
        Brightness gain. Set to -1 to use automatic adjustment.
    
    Returns
    -------
    background : np.array
        Gray scale image representing the background.

    """
    paths = glob.glob(fileLocation + '/fig_*.jpeg')
    paths = sorted(paths, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))

    images = np.ndarray([images_number, 1080, 1920], dtype=np.uint8)
    indices = np.linspace(0, len(paths) - 1, num=images_number, dtype=int)
    # indices = np.linspace(0, images_number-1, num=images_number, dtype=int)

    selected_paths = [paths[i] for i in indices]

    counter = 0
    for filename in selected_paths:
        img = cv2.imread(filename)
        # Convert the frame to grayscale
        elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)
        # DOMEgraphics.draw_image(elaborated_img, "img "+str(counter))
        images[counter] = elaborated_img
        counter += 1

    # compute background
    background = np.median(images, axis=0)
    # background = np.mean(images, axis=0)

    background = background.round().astype(np.uint8)
    background = np.squeeze(background)

    background = process_img(background, blur=DEFAULT_BLUR, gain=gain)

    DOMEgraphics.draw_image(background, "background from color " + DEFAULT_COLOR)

    return background

def make_backgrounds(experiment_names : [List, str]):
    """
    Build and save the backgound of the experiments.
    
    The images must be saved in the folder "EXPERIMENTS_DIRECTORY/EXPERIMENT_NAME/images"
    
    Parameters
    ----------
    experiment_names : List or str
        The name(s) of the experiment(s) to be tracked.
    
    Returns
    -------
    None
    
    See also
    --------
    build_background 
        Function to generate the model of the background.
    """
    if isinstance(experiment_names, str):
        experiment_names=[experiment_names]

    for exp_counter in range(len(experiment_names)):
        experiment_name = experiment_names[exp_counter]

        if os.path.isdir(os.path.join(experiments_directory, experiment_name, 'images')):
            images_folder = os.path.join(experiments_directory, experiment_name, 'images')
        else:
            images_folder = os.path.join(experiments_directory, experiment_name)


        # Build and save the background model
        background = build_background(images_folder, 25)
        cv2.imwrite(os.path.join(experiments_directory, experiment_name, 'background.jpeg'), background)
        print(f'Background of experiment {experiment_name} saved.')


def get_contours(img: np.array, bright_thresh: List, area_r: List, compactness_r: List, background_model=None,
                 expected_obj_number: int = 0, margin_factor : float = 0.25, adjustment_factor : float = 0.02,
                 plot: bool = False, verbose : bool = False ):
    """
    Perform thresholding-based object detection on an image.
    1) Image preprocessing: the image is elebaorated using DEFAULT_COLOR, DEFAULT_BLUR and AUTO_SCALING parameters.
    2) Background subtraction: the foreground is obtained by pixel-wise subtraction of the background.
        The foreground should represent only moving objects.
    3) Objects detection: objects are detected applying the given thresholds for brightness, area, and compactenss. 
    4) Compare the number of detected objects with the expected one. 
        If the number of detected objects is close to expected one the function terminates.
        If it is less than expected_obj_number (by more than margin_factor), 
        the thresholds are relaxed by adjustment_factor and the detection repeated.
        If it is greater than expected_obj_number (by more than twice margin_factor), 
        the thresholds are tightned by adjustment_factor and the detection repeated.

    Parameters
    ----------
    img : np.array
        The input image to be analyzed.
    bright_thresh : List 
        Brightness threshold for object detection. Shape=(1)
    area_r : List
        Range of acceptable area (in pixels) for object detection. Shape=(2)
    compactness_r : List
        Range of acceptable compactness values for object detection. Values in [0, 1]. Shape=(2)
    background_model : np.array = None
        An image of the background used for background subtraction. 
        If None, background subtraction is not performed.
    expected_obj_number : int = 0
        The expected number of objects in the image. 
        If the number of detected objects is not close to the expected one the thresholds are adjusted and the process repeated.
        If 0, the function will not attempt to match a specific number of objects.
    margin_factor : float = 0.25
        Margin on the difference between the number of detected and expected objects.
    adjustment_factor : float = 0.02
        Factor for adjusting the thresholds.
    plot : bool = False
        If True, the function will plot the image at each step of the processing.
    verbose : bool = False
        If True, the function will print additional details during processing.

    Returns
    -------
    contoursFiltered : List
        A list of contours for the detected objects in the image.
    
    """
    contoursFiltered = []
    if plot: DOMEgraphics.draw_image(img, "img")
    
    # Image preprocessing
    elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)
    if plot: DOMEgraphics.draw_image(elaborated_img, "elaborated img from color " + DEFAULT_COLOR)

    # Subtract the background from the frame
    if type(background_model) == type(None): background_model = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # foreground = cv2.min(cv2.subtract(elaborated_img, background_model), elaborated_img)
    foreground = cv2.subtract(elaborated_img, background_model)
    if plot: DOMEgraphics.draw_image(foreground, "foreground")

    foreground = process_img(foreground, blur=DEFAULT_BLUR)
    if plot:  DOMEgraphics.draw_image(foreground, "elaborated foreground")

    threshold = bright_thresh[0]
    a_r = area_r  # .copy()
    c_r = compactness_r  # .copy()

    first_time = True
    
    too_few_obgs = False
    too_many_obgs = False

    # Perform brightness and shape thresholding.
    # Iterate as long as the number of detected objects is not close to the expected one.
    while first_time or (too_few_obgs and threshold > 55) or (too_many_obgs and threshold < 200):
        first_time = False

        # Apply thresholding to the foreground image to create a binary mask
        ret, mask = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)
        if plot: DOMEgraphics.draw_image(mask, "mask")

        # Find contours of objects
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = img.copy()
        cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)

        # Select objects with the right size and compactness
        contoursFiltered = []
        for i in range(len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)

            # print contours info
            (Cx, Cy) = np.squeeze(get_positions(contours[i:i + 1])).astype(int)
            cv2.putText(contours_img, "A=" + str(round(area)), (Cx + 20, Cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 4)

            if a_r[0] <= area <= a_r[1]:
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * np.pi * area) / (perimeter ** 2)  # Polsby–Popper test
                cv2.putText(contours_img, "C=" + str(round(compactness, 2)), (Cx + 20, Cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

                if c_r[0] <= compactness <= c_r[1]:
                    contoursFiltered.append(contour)

        if expected_obj_number == 0: expected_obj_number = len(contoursFiltered)

        if plot: DOMEgraphics.draw_image(contours_img, "contours with thresh=" + str(threshold))


        if plot: 
            contoursFiltered_img = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(contoursFiltered_img, contoursFiltered, -1, (0, 255, 0), 3)
            DOMEgraphics.draw_image(contoursFiltered_img, "contoursFiltered with thresh=" + str(threshold))

        # If the number of detected objects is not close to the expected one adjust the thresholds and iterate
        if verbose:
            print("thresh=" + str(round(threshold)) + "\t area_r=" + str(np.around(a_r)) + "\t compactness_r=" +
                str(np.around(c_r, 2)) + "\t objects=" + str(len(contoursFiltered)) + "\t exp objects=" +
                str(expected_obj_number))

        too_many_obgs = len(contoursFiltered) - expected_obj_number > np.ceil(expected_obj_number * margin_factor * 2.0)
        too_few_obgs = len(contoursFiltered) - expected_obj_number < - np.ceil(expected_obj_number * margin_factor)

        if too_few_obgs:
            bright_thresh[0] = bright_thresh[0] * (1 - adjustment_factor)
            threshold = bright_thresh[0]
            a_r[0] = a_r[0] * (1 - adjustment_factor)
            a_r[1] = a_r[1] * (1 + adjustment_factor)
            c_r[0] = c_r[0] * (1 - adjustment_factor)
            c_r[1] = c_r[1] * (1 + adjustment_factor)

        elif too_many_obgs:
            bright_thresh[0] = bright_thresh[0] * (1 + adjustment_factor)
            threshold = bright_thresh[0]
            a_r[0] = a_r[0] * (1 + adjustment_factor)
            a_r[1] = a_r[1] * (1 - adjustment_factor)
            c_r[0] = c_r[0] * (1 + adjustment_factor)
            c_r[1] = c_r[1] * (1 - adjustment_factor)

    return contoursFiltered


def get_positions(contours : List):
    """
    Get the centers of contours resulting from image analysis

    Parameters
    ----------
    contours : List
        Contours of the detected objects. Shape=(Nx2)

    Returns
    -------
    positions : List
        Position of the center of each object. Shape=(Nx2)

    """
    positions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        positions.append([x + (w / 2), y + (h / 2)])

    return positions


def distance_from_edges(pos: List):
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
    assert (len(pos) == 2)

    dis = np.min([pos[0], pos[1], 1920 - pos[0], 1080 - pos[1]])

    assert (dis >= 0)
    return dis


def valid_positions(positions: np.array):
    """
    Get the indices of valid positions, i.e. positions in the range [0, 1920][0, 1080]

    Parameters
    ----------
    positions : np.array of float
        Array of positions. Shape=(Nx2)

    Returns
    -------
    validity : np.array
        Array of bool telling whether the corresponding position is valid.

    """
    validity0 = (positions[:, 0] >= 0) & (positions[:, 0] <= 1920)
    validity1 = (positions[:, 1] >= 0) & (positions[:, 1] <= 1080)
    validity2 = ~ np.isnan(positions[:, 0])
    validity = validity0 * validity1 * validity2

    assert len(validity) == positions.shape[0]
    return validity

def save_tracking(experiment : DOMEexp.ExperimentManager = None):
    """
    Saves tracking data, generates tracking images, and creates a video from the images.

    Parameters
    ----------
    experiment : DOMEexp.ExperimentManager = None
        The experiment whose tracking data is to be saved. 
        If None uses current_experiment.
    
    Global vars
    -----------
    current_experiment : DOMEexp.ExperimentManager
        The current experiment, used if the experiment input parameter is not given.
    positions : np.array of float
        Array of positions generated by the tracking algorithm. Shape=(MxNx2)
    inactivity : np.array of int
        Array of inactivity counters generated by the tracking algorithm. Shape=(MxN)
    total_cost : float
        Total cost resulting from the tracking procedure.
    PARAMETERS : dict
        Parameters used for the tracking.
    output_folder : str
        Name of the folder where tracking data must be saved.
    
    Returns
    -------
    None

    """
    if not experiment:
        experiment = current_experiment
    
    # generate tracking images
    overlap_trajectories(experiment)
    
    # make video from images
    output_dir = os.path.join(experiment.path, output_folder)
    DOMEgraphics.make_video(output_dir, title='tracking.mp4', fps=2, key='/trk_*.jpeg')

    # Save tracking data
    current_experiment.save_data(os.path.join(output_folder, 'analysis_data'), force=True, positions=positions,
                                     inactivity=inactivity, total_cost=total_cost, parameters=PARAMETERS)
    
    print(output_folder + ": Data, images and video updated.")

def overlap_trajectories(experiment : DOMEexp.ExperimentManager = None):
    """
    Generates tracking images by overlapping the tracked positions to the images from the experiment.
    The images are automatically saved in the experiment folder.
    
    Parameters
    ----------
    experiment : DOMEexp.ExperimentManager = None
        The experiment whose tracking images must be generated.
        If None uses current_experiment.
    
    Global vars
    -----------
    current_experiment : DOMEexp.ExperimentManager
        The current experiment, used if the experiment input parameter is not given.
    positions : np.array of float
        Array of positions generated by the tracking algorithm. Shape=(MxNx2)
    inactivity : np.array of int
        Array of inactivity counters generated by the tracking algorithm. Shape=(MxN)
    output_folder : str
        Name of the folder where tracking data must be saved.
    
    Returns
    -------
    None
    """
    if not experiment:
        experiment = current_experiment
    files = glob.glob(os.path.join(experiment.path, 'images', '*.jpeg'))
    files = sorted(files, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))

    frames_number = min(len(files), inactivity.shape[0])

    assert frames_number>0

    for counter in range(frames_number):
        filename = files[counter]
        time = DOMEexp.get_time_from_title(filename)
        img = cv2.imread(filename)
        fig = DOMEgraphics.draw_trajectories(positions[:counter + 1], [], inactivity[:counter + 1], img,
                                             title='time=' + str(time), max_inactivity=3, time_window=5, show=False)
        fig.savefig(os.path.join(experiment.path, output_folder, 'trk_' + '%04.1f' % time + '.jpeg'), dpi=100)
        print(f'\rGenerating tracking images: {round((counter+1)/frames_number*100,1)}% of {frames_number} images', end='\r')
    print("\nNow you can use DOMEgraphics.make_video to generate the video.")

def merge_trajectories(id1 : int, id2 : int):
    """
    Merge two trajectories into a single one.
    This function can be used by the user to manually correct errors in the tracking.

    Parameters
    ----------
    id1 : int
        Index of the trajectory to be updated.
    id2 : int
        Index of the trajectory to be merged into the first one.
        Must be greater than id1.
    
    Global vars
    -----------
    positions : np.array of float
        Array of positions generated by the tracking algorithm. Shape=(MxNx2)
    inactivity : np.array of int
        Array of inactivity counters generated by the tracking algorithm. Shape=(MxN)
    
    Returns
    -------
    None.

    """
    assert id1 < id2, "id2 must be greater than id1!"
    assert id1 < positions.shape[1], "id1 cannot be greater than the number of agents!"
    assert id2 < positions.shape[1], "id2 cannot be greater than the number of agents!"
    assert any(inactivity[:,id1]==0), f"Agent {id1} is never active!"
    assert any(inactivity[:,id2]==0), f"Agent {id2} is never active!"

    #print("inactivity of "+str(id1)+":\n"+ str(inactivity[:,id1]))
    #print("inactivity of "+str(id2)+":\n"+ str(inactivity[:,id2]))

    assert all(inactivity[inactivity[:,id1]==0,id2]!=0), f"Agents {id1} and {id2} are active at the same time, their trajectories cannot be merged!"

    positions[inactivity[:,id2]>=0, id1] = positions[inactivity[:,id2]>=0, id2]
    inactivity[inactivity[:,id2]>=0, id1] = inactivity[inactivity[:,id2]>=0, id2]

    positions[:,id2,:] = np.nan
    inactivity[:,id2] = -1

    #print("inactivity of "+str(id1)+" after merge:\n"+ str(inactivity[:,id1]))
    print("When you are done use save_tracking() to save updated tracking images and data.")


def extract_data_from_images(fileLocation : str, output_folder: str, parameters : dict, background : np.ndarray = None, 
                             activation_times : List = [], terminal_time : float = -1, verbose:bool = False, show:bool = True):
    """
    Performs objects detection and tracking using images saved in a given folder.
    Images are sorted according to the file name and analysed sequentially.
    For each image objects detection and IDs matching are performed.
    Then the objects' positions at the next time step are estimated.

    Parameters
    ----------
    fileLocation : str
        Path of the folder containing the images.
    output_folder : str
        Name of the folder where the tracking output will be stored.
    parameters : dict
        The parameters for objects detection and tracking.
        parameters = {
            "AREA_RANGE"      : [a_min, a_max],   # range of area for obj detection, positive values [pixels]
            "COMPAC_RANGE"    : [c_min, c_max],   # range of compactness for obj detection, values in [0,1]
            "BRIGHT_THRESH"   : [brightness_min], # brightness threshold used for object detection, values in [0, 255]
            "TYPICAL_VEL"     : typical_velocity, # coeff used to scale the id assignment costs, positive values[px/s]
            "INERTIA"         : inertia_coeff     # coeff used for position estimation, values in [0,1]
        }
    background : np.ndarray = None
        The background image generated using the build_background function.
        If None background subtraction is not performed.
    activation_times : List = []
        The sampling time instants. 
        If empty uniform sampling step is assumed.
    terminal_time : float = -1
        Time to terminate the tracking. If negative, tracking is performed on the whole experiment.
    verbose : bool = False
        If True, prints additional information during execution.
    show : bool = True
        If True, shows the tracking image at each time step.
    
    Returns
    -------
    positions : np.array of float
        Array of positions generated by the tracking algorithm. 
        Nans will appear when the corresponding agent is not detected.
        Shape=(MxNx2) where M is number of time instants and N the number of detected objects.
    inactivity : np.array of int
        Array of inactivity counters generated by the tracking algorithm. 
        The inactivity counter indicates the number of time steps from last time the object was detected.
        Currently detected objects have 0 inactivity.
        Objects before their first detection have negative inactivity.
        Objects that have been detected and then lost have positive inactivity.
        Shape=(MxN) where M is number of time instants and N the number of detected objects.
    total_cost : float
        Total cumulative cost of the tracking procedure.
    """
    files = glob.glob(fileLocation + '/*.jpeg')
    files = sorted(files, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))

    # if terminal_time is negative perform tracking on the whole experiment
    # otherwise cut out excess images
    if terminal_time < 0:
        terminal_time = DOMEexp.get_time_from_title(files[-1])
    else:
        files = [f for f in files if DOMEexp.get_time_from_title(f) <= terminal_time]

    frames_number = len(files)
    number_of_objects = 0
    n_detected_objects = 0
    total_cost = 0

    contours = []
    positions = np.empty([frames_number, 0, 2], dtype=float) * np.nan
    inactivity = - np.ones([frames_number, 0], dtype=int)

    time = 0.0
    counter = 0

    bright_thresh = parameters["BRIGHT_THRESH"].copy()
    area_r = parameters["AREA_RANGE"].copy()
    compactness_r = parameters["COMPAC_RANGE"].copy()
    
    if len(activation_times) == 0:
        activation_times = np.linspace(0, terminal_time, len(files))

    print("Performing detection and tracking...")
    while time < terminal_time and counter < len(files):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = DOMEexp.get_time_from_title(filename)

        print('\rTracking: t = ' + str(time) + f's (total time = {terminal_time}s)', end='\r')
        if verbose: print()

        # collect contours and positions from new image
        plot_detection_steps = counter == 0
        new_contours = get_contours(img, bright_thresh, area_r, compactness_r, background, n_detected_objects,
                                    plot=plot_detection_steps, verbose=verbose)
        new_positions = get_positions(new_contours)
        n_detected_objects = len(new_positions)

        # on first iteration assign new ids to all agents
        if counter == 0:
            deltaT = activation_times[counter+1]-activation_times[counter]
            new_ids = list(range(0, n_detected_objects))

        # on following iterations perform tracking
        else:
            deltaT = activation_times[counter]-activation_times[counter-1]
            est_positions = positions[counter]  # select positions at previous time instant
            est_positions = est_positions[valid_positions(est_positions)]  # select valid positions
            new_ids, cost = agentMatching(new_positions, est_positions, inactivity[counter - 1], deltaT, verbose)
            total_cost += cost

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
                with np.errstate(invalid='ignore'):
                    empty_row = np.empty([frames_number, 1, 2], dtype=float) * np.nan
                positions = np.concatenate([positions, empty_row], axis=1)
                positions[counter, number_of_objects] = new_positions[new_ids.index(new_id)]

                empty_row = - np.ones([frames_number, 1], dtype=int)
                inactivity = np.concatenate([inactivity, empty_row], axis=1)
                inactivity[counter, number_of_objects] = 0

                contours.append(new_contours[new_ids.index(new_id)])
                number_of_objects += 1

        # for lost objects estimate position and increase inactivity
        for lost_id in lost_obj_ids:
            inactivity[counter, lost_id] = inactivity[counter - 1, lost_id] + 1

        # estimate velocities and future positions
        up_to_now_positions = positions[0:counter + 1]  # select positions up to current time instant
        velocities = estimate_velocities(up_to_now_positions, deltaT)
        if counter < frames_number - 1:
            deltaT_next = activation_times[counter+1]-activation_times[counter]
            positions[counter + 1] = estimate_positions(positions[counter], velocities, deltaT_next)

        # check data integrity
        assert all(valid_positions(positions[counter]))

        # print image
        fig = DOMEgraphics.draw_trajectories(positions[:counter + 1], [], inactivity[:counter + 1], img,
                                             title='time=' + str(time), max_inactivity=3, time_window=5, show=show)
        fig.savefig(os.path.join(fileLocation, output_folder, 'trk_' + '%04.1f' % time + '.jpeg'), dpi=100)

        # if verbose print info
        if verbose:
            print('deltaT = ' + str(round(deltaT,3)) + 's')
            print('total number of objects = ' + str(number_of_objects))
            print('detected objects = ' + str(n_detected_objects))
            print('new ids = ' + str(newly_allocated_ids) + '\t tot = ' + str(len(newly_allocated_ids)))
            print('total lost ids = ' + str(len(lost_obj_ids)), end='\n\n')

        counter+=1

    # print average info
    print('\nTotal number of detected objects = ' + str(inactivity.shape[1]))
    print('Total matching cost = ' + str(round(total_cost,2)), end='\n\n')

    return positions, inactivity, total_cost

def load_tracking(tracking_name : str = None, experiment : [str, DOMEexp.ExperimentManager] = None):    
    """
    Load data from an existing tracking file ('analysis_data.npz').
    Use this function if you need to analyze or correct the tracking data.
    
    Parameters
    ----------
    tracking_name : str = None
        Name of the tracking folder where tracking data are saved.
        If None the global var output_folder is used.
    experiment : [str, DOMEexp.ExperimentManager] = None
        Name or ExperimentManager of the experiment whose tracking will be loaded.
        
    Returns
    -------
    positions : np.array of float
        Array of positions generated by the tracking algorithm. 
        Nans will appear when the corresponding agent is not detected.
        Shape=(MxNx2) where M is number of time instants and N the number of detected objects.
    inactivity : np.array of int
        Array of inactivity counters generated by the tracking algorithm. 
        The inactivity counter indicates the number of time steps from last time the object was detected.
        Currently detected objects have 0 inactivity.
        Objects before their first detection have negative inactivity.
        Objects that have been detected and then lost have positive inactivity.
        Shape=(MxN) where M is number of time instants and N the number of detected objects.
    total_cost : float
        Total cumulative cost of the tracking procedure.
    PARAMETERS : dict
        The parameters used in the given tracking.
        parameters = {
            "AREA_RANGE"      : [a_min, a_max],   # range of area for obj detection, positive values [pixels]
            "COMPAC_RANGE"    : [c_min, c_max],   # range of compactness for obj detection, values in [0,1]
            "BRIGHT_THRESH"   : [brightness_min], # brightness threshold used for object detection, values in [0, 255]
            "TYPICAL_VEL"     : typical_velocity, # coeff used to scale the id assignment costs, positive values[px/s]
            "INERTIA"         : inertia_coeff     # coeff used for position estimation, values in [0,1]
        }
    current_experiment : DOMEexp.ExperimentManager
        ExperimentManager of the loaded experiment.
    
    See also
    --------
    start_tracking 
        Function to execute the tracking.
    """
    global positions, inactivity, total_cost, parameters, current_experiment

    # set current experiment
    if isinstance(experiment, str):
        current_experiment = DOMEexp.open_experiment(experiment, experiments_directory)
    elif isinstance(experiment, DOMEexp.ExperimentManager):
        current_experiment = experiment

    # get tracking name
    if not tracking_name:
        tracking_name = output_folder

    with current_experiment.get_data(os.path.join(tracking_name, 'analysis_data.npz'), allow_pickle=True) as data:
        positions = data['positions']
        inactivity = data['inactivity']

        if 'total_cost' in data.files:
            total_cost = data['total_cost'].item()

        if 'parameters' in data.files:
            PARAMETERS = data['parameters'].item()

    assert inactivity.shape[0] == positions.shape[0]
    assert inactivity.shape[1] == positions.shape[1]
    print(f'{tracking_name} loaded.\nTotal cost = {np.round(total_cost,2)}, total objects = {inactivity.shape[1]}, time frames = {inactivity.shape[0]}')

    return positions, inactivity, total_cost, PARAMETERS, current_experiment

def start_tracking(experiment_names : [List, str]):
    """
    Perform tracking of the given experiment(s).
    For each experiment:
    1) Builds the background model
    2) Perform tracking and generates tracking images
    3) Makes a video from the tracking images
    4) Saves the resulting tracking data in the analysis_data.npz file.
    
    The images must be saved in the folder "EXPERIMENTS_DIRECTORY/EXPERIMENT_NAME/images"
    
    Parameters
    ----------
    experiment_names : List or str
        The name(s) of the experiment(s) to be tracked.
    
    Returns
    -------
    None
    
    See also
    --------
    build_background 
        Function to generate the model of the background.
    extract_data_from_images
        Function to perform the tracking.
    """
    if isinstance(experiment_names, str):
        experiment_names=[experiment_names]

    for exp_counter in range(len(experiment_names)):
        print(f'\nTracking experiment {exp_counter+1} of {len(experiment_names)}')
        experiment_name = experiment_names[exp_counter]
        current_experiment = DOMEexp.open_experiment(experiment_name, experiments_directory)

        with current_experiment.get_data('data.npz') as data:
            activation_times = data['activation_times']

        if os.path.isdir(os.path.join(experiments_directory, experiment_name, 'images')):
            images_folder = os.path.join(experiments_directory, experiment_name, 'images')
        else:
            images_folder = os.path.join(experiments_directory, experiment_name)

        output_dir = os.path.join(experiments_directory, experiment_name, output_folder)
        try:
            os.mkdir(output_dir)
        except OSError:
            pass

        # Build background model
        print("Building the background model...")
        background = build_background(images_folder, 25)
        cv2.imwrite(os.path.join(experiments_directory, experiment_name, output_folder, 'background.jpeg'), background)

        # extract data and generate tracking images
        positions, inactivity, total_cost = extract_data_from_images(images_folder, output_dir, PARAMETERS,
                                        background, activation_times, terminal_time, verbose, show_tracking_images)

        # make video from images
        DOMEgraphics.make_video(output_dir, title='tracking.mp4', fps=2, key='/trk_*.jpeg')

        # Save tracking data
        current_experiment.save_data(os.path.join(output_folder, 'analysis_data'), force=True, positions=positions,
                                         inactivity=inactivity, total_cost=total_cost, parameters=PARAMETERS)

# MAIN -----------------------------------------------------------------------
if __name__ == '__main__':
    # IMAGE POROCESSING PARAMETERS
    AUTO_SCALING = -1       # value for automatic brightness adjustment
    DEFAULT_COLOR = "red"   # color channel used for gray-scale conversion "gray", "blue", "green" or "red"
    DEFAULT_BLUR = 9        # size of the blurring window [in pixels]


    # OBJECT DETECTION AND TRACKING PARAMETERS
    # PARAMETERS = {
    #     "AREA_RANGE"      : [a_min, a_max],   # range of area for obj detection, positive values [pixels]
    #     "COMPAC_RANGE"    : [c_min, c_max],   # range of compactness for obj detection, values in [0,1]
    #     "BRIGHT_THRESH"   : [brightness_min], # brightness threshold used for object detection, values in [0, 255]
    #     "TYPICAL_VEL"     : typical_velocity, # coeff used to scale the id assignment costs, positive values[px/s]
    #     "INERTIA"         : inertia_coeff     # coeff used for position estimation, values in [0,1]
    # }
    
    
    # Euglena
    PARAMETERS = {
        "AREA_RANGE" : [175, 1500],
        "COMPAC_RANGE" : [0.55, 0.90],
        "BRIGHT_THRESH" : [85],
        "TYPICAL_VEL" : 70,             # [px/s]
        "INERTIA" : 0.9
    }

    # # P. Caudatum
    # PARAMETERS = {
    #     "AREA_RANGE" : [250, 3000],
    #     "COMPAC_RANGE" : [0.5, 0.9],
    #     "BRIGHT_THRESH" : [70],
    #     "TYPICAL_VEL" : 100
    #     "INERTIA" : 0.9
    # }

    # # # P. Bursaria
    # PARAMETERS = {
    #     "AREA_RANGE" : [150, 1500],
    #     "COMPAC_RANGE" : [0.6, 0.9],
    #     "BRIGHT_THRESH" : [60],
    #     "TYPICAL_VEL" : 50
    #     "INERTIA" : 0.9
    # }

    # # Volvox
    # PARAMETERS = {
    #     "AREA_RANGE" : [1000, 6000],
    #     "COMPAC_RANGE" : [0.7, 1.0],
    #     "BRIGHT_THRESH" : [70],
    #     "TYPICAL_VEL" : 30
    #     "INERTIA" : 0.9
    # }

    # Directory where DOME experiments folders are saved
    # experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
    experiments_directory = '/Volumes/DOMEPEN/Experiments'
    # experiments_directory = 'D:\AndreaG_DATA\Experiments'


    # tracked_experiments = ["2023_06_15_Euglena_1","2023_06_15_Euglena_2",
    #                       "2023_06_26_Euglena_13", "2023_06_26_Euglena_23", 
    #                       "2023_06_26_Euglena_37","2023_07_10_Euglena_5","2023_07_10_Euglena_6", 
    #                       "2023_07_10_Euglena_8","2023_07_10_Euglena_10","2023_07_10_Euglena_12",
    #                       "2023_07_10_Euglena_15","2023_06_15_Euglena_10",
    #                       "2023_06_15_Euglena_11","2023_06_26_Euglena_28","2023_07_10_Euglena_19"
    #                       "2023_06_26_Euglena_36", "2023_06_26_Euglena_37", "2023_06_15_Euglena_16", 
    #                       "2023_07_10_Euglena_21", "2023_07_10_Euglena_22", "2023_06_26_Euglena_39"]

    experiments_switch_10s = ['2023_06_15_Euglena_7','2023_06_26_Euglena_23','2023_06_26_Euglena_24',
                              '2023_07_10_Euglena_15','2023_07_10_Euglena_16'];
    
    experiments_switch_5s = ['2023_06_15_Euglena_8','2023_06_15_Euglena_9','2023_06_15_Euglena_10',
                              '2023_06_26_Euglena_25','2023_06_26_Euglena_26',
                              '2023_07_10_Euglena_17','2023_07_10_Euglena_18'];
    
    experiments_switch_1s = ['2023_06_15_Euglena_11','2023_06_26_Euglena_27','2023_06_26_Euglena_28',
                              '2023_07_10_Euglena_19','2023_07_10_Euglena_20'];
    
    experiments_ramp = ['2023_06_15_Euglena_5','2023_06_15_Euglena_6',
                        '2023_06_26_Euglena_22','2023_06_26_Euglena_21',
                        '2023_07_10_Euglena_13','2023_07_10_Euglena_14'];

    experiments_gradient_central_light = ['2023_06_12_Euglena_3','2023_06_12_Euglena_4','2023_06_14_Euglena_7',
                                          '2023_06_15_Euglena_14','2023_06_23_Euglena_5','2023_06_23_Euglena_6',
                                          '2023_06_26_Euglena_5','2023_06_26_Euglena_6','2023_06_26_Euglena_33'];
    
    experiments_gradient_central_dark = ['2023_06_14_Euglena_10','2023_06_15_Euglena_15','2023_06_23_Euglena_7',
                                          '2023_06_23_Euglena_8','2023_06_23_Euglena_9','2023_06_26_Euglena_7',
                                          '2023_06_26_Euglena_8','2023_06_26_Euglena_34','2023_06_26_Euglena_35',
                                          '2023_07_10_Euglena_23','2023_07_10_Euglena_24'] 
    
    experiments_gradient_lateral = ['2023_06_12_Euglena_5','2023_06_13_Euglena_16','2023_06_14_Euglena_8',
                                    '2023_06_15_Euglena_13','2023_06_23_Euglena_3','2023_06_23_Euglena_4',
                                    '2023_06_26_Euglena_3','2023_06_26_Euglena_4','2023_06_26_Euglena_31',
                                    '2023_06_26_Euglena_32']
    
    experiments_half_half = ['2023_06_12_Euglena_2','2023_06_14_Euglena_6','2023_06_15_Euglena_12',
                             '2023_06_26_Euglena_29','2023_06_26_Euglena_30','2023_06_23_Euglena_1',
                             '2023_06_23_Euglena_2','2023_06_26_Euglena_2','2023_06_26_Euglena_1']
    
    experiments_circle_light = ['2023_06_12_Euglena_1','2023_06_14_Euglena_1','2023_06_15_Euglena_16','2023_06_23_Euglena_10',
                                '2023_06_23_Euglena_11','2023_06_26_Euglena_9','2023_06_26_Euglena_10','2023_06_26_Euglena_36',
                                '2023_06_26_Euglena_37','2023_07_10_Euglena_26']
    
    experiments_circle_dark = ['2023_06_13_Euglena_6','2023_06_13_Euglena_15','2023_06_15_Euglena_17',
                               '2023_06_15_Euglena_18','2023_06_23_Euglena_12','2023_06_23_Euglena_13',
                               '2023_06_26_Euglena_11','2023_06_26_Euglena_12','2023_06_26_Euglena_38',
                               '2023_06_26_Euglena_39','2023_07_10_Euglena_25','2023_07_10_Euglena_22']
    
    experiments_BCL = ['2023_07_10_Euglena_30','2023_07_10_Euglena_34','2023_07_10_Euglena_35',
                              '2023_07_10_Euglena_36','2023_07_10_Euglena_37','2023_07_10_Euglena_38']
    
    

    # Name of the experiment(s) to be tracked
    experiment_names = ["2023_07_10_Euglena_26"]
    
    # Name of the folder to save tracking results
    output_folder = 'tracking_' + datetime.today().strftime('%Y_%m_%d')
    #output_folder = 'tracking_test'

    # Tracking options
    terminal_time = -1          # time to stop tracking [s], set negative to track the whole experiment
    verbose = False             # print info during tracking
    show_tracking_images = False # print images during tracking
    #show_tracking_images = os.name == 'posix' # print images during tracking

    # Useful commands
    print('Now use one of the following commands:'
          '\n\tstart_tracking(experiment_names)\t\t\t\t\t\t\t\t\t\t\t\t\tStart tracking of the given experiment(s).'
          '\n\ttest_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)\tTest obj detection on a random image in images_folder.'
          '\n\ttest_detection_parameters(image_name, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)\t\tTest obj detection on the selected image.'
          '\n\tpositions, inactivity, total_cost, PARAMETERS, current_experiment=load_tracking(output_folder,experiment_name)\tLoad data from an existing tracking.'
          '\n\tmake_backgrounds(experiment_names)\t\t\t\t\t\t\t\t\t\t\t\tBuild and save background models of the given experiment(s).')

    # test thresholds for object detection
    # test_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)

    # start tracking
    # start_tracking(experiment_names)

    # load existing tracking data
    # load_tracking(experiment_name : str)
