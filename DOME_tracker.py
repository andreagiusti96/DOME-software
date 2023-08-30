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
    cost = (distance * DISTANCE_COST_FACTORS[0] + distance ** 2 * DISTANCE_COST_FACTORS[1]) / (
                inactivity ** 2 * 0.25 + 1)
    cost += inactivity * INACTIVITY_COST_FACTORS[0] + inactivity ** 2 * INACTIVITY_COST_FACTORS[1]
    return cost


def plotCosts():
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
    plt.xlabel('distance/TYPICAL_D')


def agentMatching(new_positions: np.array, positions: np.array, inactivity: List):
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
    new_positions = np.array(new_positions)
    number_of_objects = sum(valid_positions(positions))
    costs_matching = np.ndarray([len(new_positions), number_of_objects])
    costs_newid = np.ndarray([len(new_positions), len(new_positions)])

    distances = np.squeeze(scipy.spatial.distance.cdist(new_positions, positions))
    distances = distances / TYPICAL_D

    # build the matrix of costs
    for i in range(positions.shape[0]):
        # distances = np.squeeze(scipy.spatial.distance.cdist(new_positions, positions[i,:]))
        # costs_matching[i,:] = distances[i,:]*DISTANCE_COST_FACTORS[0] +distances[i,:]**2*DISTANCE_COST_FACTORS[1]
        # inactivity_cost = (np.array(inactivity)) * INACTIVITY_COST_FACTORS[0] + (np.array(inactivity)**2) * INACTIVITY_COST_FACTORS[1]
        # costs_matching[i,:] += inactivity_cost
        costs_matching[:, i] = matchingCost(distances[:, i], inactivity[i])

    for i in range(new_positions.shape[0]):
        cost_newid = np.min(
            [distance_from_edges(new_positions[i]) / TYPICAL_D, NEW_ID_COST_DIST_CAP]) ** 2 + NEW_ID_COST_MIN
        costs_newid[i, :] = np.ones([len(new_positions)]) * cost_newid

    costs = np.concatenate((costs_matching, costs_newid), axis=1)

    # Hungarian optimization algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    cost = costs[row_ind, col_ind].sum()

    # update ids
    new_ids = [i for i in col_ind]

    print('matching cost = ' + str(round(cost, 2)) + '\t avg = ' + str(round(cost / (len(new_ids) + 0.001), 2)))

    return new_ids


def estimate_velocities(positions: np.array):
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

    if positions.shape[0] >= 2:
        valid_pos_idx = valid_positions(positions[-2])
        velocities[valid_pos_idx] = positions[-1, valid_pos_idx] - positions[-2, valid_pos_idx]

    speeds = np.linalg.norm(velocities, axis=1)

    # print("avg speed = " + str(round(np.mean(speeds),1)) + "\tmax = " + str(round(max(speeds),1)) + "\tid =" + str(np.argmax(speeds)))

    assert velocities.shape[1] == 2
    return velocities


def estimate_positions(old_pos: np.array, velocity: np.array):
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

    non_valid_pos_idx = ~ valid_positions(estimated_pos)
    estimated_pos[non_valid_pos_idx] = estimated_pos[non_valid_pos_idx] - velocity[non_valid_pos_idx] * inertia

    return estimated_pos


def interpolate_positions(positions: np.array):
    interpolated_pos = positions.copy()

    for obj in range(positions.shape[1]):
        first_index = (~np.isnan(positions[:, obj, 0])).argmax(0)
        last_index = positions.shape[0] - (~np.isnan(positions[:, obj, 0]))[::-1].argmax(0) - 1

        nans = np.isnan(positions[first_index:last_index + 1, obj, 0])
        missing_points = np.where(nans)[0] + first_index
        valid_points = np.where(~nans)[0] + first_index

        if len(missing_points) > 0:
            trajectory_x = positions[valid_points, obj, 0]
            trajectory_y = positions[valid_points, obj, 1]
            interpolated_pos[missing_points, obj, 0] = np.interp(missing_points, valid_points, trajectory_x)
            interpolated_pos[missing_points, obj, 1] = np.interp(missing_points, valid_points, trajectory_y)

        # print(np.concatenate([positions[:last_index+2,obj], interpolated_pos[:last_index+2,obj]], axis=1))
    return interpolated_pos


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
                 expected_obj_number: int = 0, plot: bool = False):
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
        print("thresh=" + str(round(threshold)) + "\t area_r=" + str(np.around(a_r)) + "\t compactness_r=" + str(
            np.around(c_r, 2)) + "\t objects=" + str(len(contoursFiltered)) + "\t exp objects=" + str(
            expected_obj_number))
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
        Contours of the detected objects. (Shape=Nx2)

    Returns
    -------
    positions : List
        Position of the center of each object. (Shape=Nx2)

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


def extract_data_from_images(fileLocation, background: np.ndarray, bright_thresh: List, area_r: List,
                             compactness_r: List, output_folder: str, terminal_time : float = -1):
    files = glob.glob(fileLocation + '/*.jpeg')
    files = sorted(files, key=lambda x: float(re.findall("(\d+.\d+)", x)[-1]))
    
    if terminal_time < 0:
        terminal_time = DOMEexp.get_time_from_title(files[-1])

    frames_number = len(files)
    number_of_objects = 0
    n_detected_objects = 0

    contours = [];
    positions = np.empty([frames_number, 0, 2], dtype=float) * np.nan;
    inactivity = - np.ones([frames_number, 0], dtype=int);
    
    time = 0.0
    counter = 0
    
    print("Performing detection and tracking...")
    while time < terminal_time and counter < len(files):
    #for counter in range(len(files)):
        # for counter in range(10):
        # declare vars
        filename = files[counter]
        img = cv2.imread(filename)
        time = DOMEexp.get_time_from_title(filename)
        print('t = ' + str(time))

        # collect contours and positions from new image
        plot_detection_steps = counter == 0
        new_contours = get_contours(img, bright_thresh, area_r, compactness_r, background, n_detected_objects,
                                    plot_detection_steps)
        new_positions = get_positions(new_contours)
        n_detected_objects = len(new_positions)

        # on first iteration assign new ids to all agents
        if counter == 0:
            new_ids = list(range(0, n_detected_objects))

        # on following iterations perform tracking
        else:
            est_positions = positions[counter]  # select positions at previous time instant
            est_positions = est_positions[valid_positions(est_positions)]  # select valid positions
            new_ids = agentMatching(new_positions, est_positions, inactivity[counter - 1])

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
                empty_row = - np.empty([frames_number, 1, 2], dtype=float) * np.nan
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
        velocities = estimate_velocities(up_to_now_positions)
        if counter < frames_number - 1:
            positions[counter + 1] = estimate_positions(positions[counter], velocities)

        # check data integrity
        assert all(valid_positions(positions[counter]))

        # print image
        fig = DOMEgraphics.draw_trajectories(positions[:counter + 1], [], inactivity[:counter + 1], img,
                                             title='time=' + str(time), max_inactivity=3, time_window=5, show=False)
        fig.savefig(os.path.join(fileLocation, output_folder, 'trk_' + '%04.1f' % time + '.jpeg'), dpi=100)

        # print info
        print('total number of objects = ' + str(number_of_objects))
        print('detected objects = ' + str(n_detected_objects))
        print('new ids = ' + str(newly_allocated_ids) + '\t tot = ' + str(len(newly_allocated_ids)))
        print('total lost ids = ' + str(len(lost_obj_ids)) + '\n')
        
        counter+=1

    return positions, inactivity


# MAIN
if __name__ == '__main__':
    # CONSTANTS
    DEFAULT_COLOR = "red"
    DEFAULT_BLUR = 9
    AUTO_SCALING = -1

    # Euglena
    AREA_RANGE = [250, 3000]; COMPAC_RANGE = [0.6, 0.9]; BRIGHT_THRESH = [85]
    TYPICAL_D = 25

    # P. Caudatum
    # AREA_RANGE = [250, 3000]; COMPAC_RANGE = [0.5, 0.9]; BRIGHT_THRESH = [70]
    # TYPICAL_D = 50

    # # P. Bursaria
    # AREA_RANGE = [150, 1500];
    # COMPAC_RANGE = [0.6, 0.9];
    # BRIGHT_THRESH = [60]
    # TYPICAL_D = 25

    # Volvox
    # AREA_RANGE = [1000, 6000]; COMPAC_RANGE = [0.7, 1.0]; BRIGHT_THRESH = [70]
    # TYPICAL_D = 15

    # experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
    # experiments_directory = '\\\\tsclient\DOMEPEN\Experiments'
    experiments_directory = '/Volumes/DOMEPEN/Experiments'
    #experiments_directory = 'D:\AndreaG_DATA\Experiments'
    experiment_name = "2023_06_26_Euglena_37"
    output_folder = 'tracking_prova'
    
    terminal_time = -1;

    current_experiment = DOMEexp.open_experiment(experiment_name, experiments_directory)

    if os.path.isdir(os.path.join(experiments_directory, experiment_name, 'images')):
        images_folder = os.path.join(experiments_directory, experiment_name, 'images')
    else:
        images_folder = os.path.join(experiments_directory, experiment_name)

    # test_detection_parameters(images_folder, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE)

    output_dir = os.path.join(experiments_directory, experiment_name, output_folder)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    # Build background model
    print("Building the background model...")
    background = build_background(images_folder, 25)
    cv2.imwrite(os.path.join(experiments_directory, experiment_name, output_folder, 'background.jpeg'), background)

    # extract data
    positions, inactivity = extract_data_from_images(images_folder, background, BRIGHT_THRESH, AREA_RANGE, COMPAC_RANGE,
                                                     output_dir, terminal_time)

    # make video from images
    DOMEgraphics.make_video(output_dir, title='tracking.mp4', fps=2)

    # Save tracking data
    analised_data_path = os.path.join(output_dir, 'analysis_data.npz')
    current_experiment.save_data(os.path.join(output_folder, 'analysis_data'), positions=positions,
                                     inactivity=inactivity)
