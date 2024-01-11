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
import os
import matplotlib.pyplot as plt
import random
from datetime import datetime
from typing import List


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
    new_positions : np.array Shape=(Nx2)
        Positions of detected objects.
    positions : np.array
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
    positions : np.array
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
    positions : np.array
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


def test_detection_parameters(fileLocation, bright_thresh, area_r: List, compactness_r: List):
    if os.path.isdir(fileLocation):
        files = glob.glob(fileLocation + '/*.jpeg')
        files = sorted(files, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))
        filename = random.choice(files)
    elif os.path.isfile(fileLocation):
        filename = fileLocation
        fileLocation = os.path.dirname(fileLocation)
    else:
        raise (Exception(f'Not a file or a directory: {fileLocation}'))

    img = cv2.imread(filename)

    background = build_background(fileLocation, 25)

    new_contours = get_contours(img, bright_thresh, area_r, compactness_r, background, 0, True)


def process_img(img: np.array, color: str = "", blur: int = 0, gain: float = 1., contrast: bool = False,
                equalize: bool = False, plot: bool = False):
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


def build_background(fileLocation: str, images_number: int, gain: float = 1.0):
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


def get_contours(img: np.array, bright_thresh: List, area_r: List, compactness_r: List, background_model=None,
                 expected_obj_number: int = 0, plot: bool = False, verbose : bool = False):
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
    contoursFiltered = []
    if plot: DOMEgraphics.draw_image(img, "img")

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
    margin_factor = 0.25
    adjustment_factor = 0.02

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

        contoursFiltered_img = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursFiltered_img, contoursFiltered, -1, (0, 255, 0), 3)
        if plot: DOMEgraphics.draw_image(contoursFiltered_img, "contoursFiltered with thresh=" + str(threshold))

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


def get_positions(contours):
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
    positions : np.array
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


def extract_data_from_images(fileLocation, background: np.ndarray, parameters : dict,
                             output_folder: str, activation_times : List = [],
                             terminal_time : float = -1, verbose:bool = False, show:bool = True):
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

    print("Performing detection and tracking...")
    while time < terminal_time and counter < len(files):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = DOMEexp.get_time_from_title(filename)

        print('\rTracking: t = ' + str(time) + f' (total time = {terminal_time})', end='\r')
        if verbose: print()

        # collect contours and positions from new image
        plot_detection_steps = counter == 0
        new_contours = get_contours(img, bright_thresh, area_r, compactness_r, background, n_detected_objects,
                                    plot_detection_steps, verbose)
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

        # print info
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

def start_tracking(experiment_names : [List, str]):
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
        positions, inactivity, total_cost = extract_data_from_images(images_folder, background, PARAMETERS,
                                        output_dir, activation_times, terminal_time, verbose, show_tracking_images)

        # make video from images
        DOMEgraphics.make_video(output_dir, title='tracking.mp4', fps=2, key='/trk_*.jpeg')

        # Save tracking data
        analised_data_path = os.path.join(output_dir, 'analysis_data.npz')
        current_experiment.save_data(os.path.join(output_folder, 'analysis_data'), force=True, positions=positions,
                                         inactivity=inactivity, total_cost=total_cost, parameters=PARAMETERS)



def load_tracking(tracking_name : str = None, experiment : [str, DOMEexp.ExperimentManager] = None):    
    
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

# MAIN -----------------------------------------------------------------------
if __name__ == '__main__':
    # CONSTANTS
    AUTO_SCALING = -1       # value for automatic brightness adjustment

    
    # IMAGE POROCESSING PARAMETERS
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
    #                       "2023_06_15_Euglena_11","2023_06_26_Euglena_28","2023_07_10_Euglena_19"]

    # Name of the experiment(s) to be tracked
    experiment_names = ["2023_06_15_Euglena_1"]
    
    # Name of the folder to save tracking results
    #output_folder = 'tracking_' + datetime.today().strftime('%Y_%m_%d')
    output_folder = 'tracking_test'

    # Tracking options
    terminal_time = -1          # time to stop tracking [s], set negative to track the whole experiment
    verbose = False             # print info during tracking
    show_tracking_images = True # print images during tracking
    #show_tracking_images = os.name == 'posix' # print images during tracking

    # Useful commands
    print('Now use one of the following commands:'
          '\n\tstart_tracking(experiment_names)\t\t\t\t\t\t\t\t\t\t\t\t\tStart tracking of the given experiment(s).'
          '\n\ttest_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)\tTest obj detection on a random image in images_folder.'
          '\n\ttest_detection_parameters(image_name, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)\t\tTest obj detection on the selected image.'
          '\n\tpositions, inactivity, total_cost, PARAMETERS, current_experiment=load_tracking(output_folder,experiment_name)\tLoad data from an existing tracking.')

    # test thresholds for object detection
    # test_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)

    # start tracking
    # start_tracking(experiment_names)

    # load existing tracking data
    # load_tracking(experiment_name : str)
