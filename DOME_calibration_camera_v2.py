# This code is provided to guide users through the process of calibration to find a mathematical
# transformation that describes the mapping of camera pixels to projector pixels in the DOME. The
# DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio Denniss. This code requires
# the "DOME_caibration_projector.py" file to be run in parallel on the Raspberry Pi 0 connected to
# the DOME projector. To calibrate the DOME, run this file and follow the on screen instructions.
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import DOME_communication as DOMEcomm
import DOME_imaging_utilities as DOMEutil
import DOME_transformation as DOMEtran

import numpy as np
import cv2
import json
import time
import os
from datetime import datetime
from datetime import date

class SettingsFileFormatError(Exception):
    '''
    Exception class for handling errors raised when calibration settings cannot be read from a
    file. The file should specify values for "brightness", "threshold", "region size" and
    "scan increment".
    '''
    
    def __init__(self, key : str):
        '''
        Sets up the parent class for exceptions and provides an error message.
        ---
        Parameters
            key : str
                The settings dictionary key that is missing from the json file.
        '''
        self.error_message = f'Format of calibration settings file is not recognised.\n Setting ' \
                             f'key "{key}" not specified.'
        super().__init__(self.error_message)
    
    def print_error_message(self):
        '''
        Prints the error message without interupting execution.
        '''
        print(self.error_message)


def load_settings(file_name : str, keys : list):
    '''
    Reads calibration setting values from a json file.
    ---
    Parameters
        file_name : str
            Name of json file to read settings from.
    ---
    Outputs
        stored_settings : dict
            Dictionary of calibration settings.
    '''
    with open(file_name, 'r') as file:
        stored_settings = json.load(file)
    for setting in keys:
        # Check that the file contains all of the expected parameters.
        if not setting in stored_settings.keys():
            raise SettingsFileFormatError(setting)
    return stored_settings

def custom_input(message : str, step : int):
    '''
    Augmented version of input command to facilitate adjustments to variable representing
    progression through calibration procedure {step}: inputting "back" reduces value by 1;
    "restart" resets value back to 0.
    ---
    Parameters
        message : str
            Message to display when user input is requested.
        step : int
            Current stage of calibration procedure.
    ---
    Outputs
        user_input_segments : list[str,...]
            User input split by space character " ".
        new_step
            New stage of calibration procedure.
    '''
    user_input = input(message)
    user_input_segments = ['']
    if user_input == 'skip':
        new_step = 7
    elif user_input == 'next':
        new_step = step + 1
    elif user_input == 'back':
        new_step = max(step - 1, 1)
    elif user_input == 'restart':
        new_step = 1
    elif user_input == 'exit':
        new_step = 0
    else:
        user_input_segments = user_input.split(' ')
        new_step = step
    return user_input_segments, new_step

def overlay_grid(image : np.ndarray, spacing : int, thickness : int, colour : tuple):
    '''
    Draw a grid pattern over an image.
    ---
    Parameters
        image : np.ndarray
            Original image.
        spacing : int
            Separation between the start of grid lines in number of pixels.
        thickness : int
            Thickness of grid lines in number of pixels.
        colour : tuple[int, int, int]
            Colour of grid lines, specified in BGR order.
    ---
    Outputs
        image_with_grid : np.ndarray
            Original image with grid drawn over it.
    '''
    image_with_grid = image.copy()
    for c in range(0, len(colour)):
        for s in range(1, int(np.ceil(image.shape[0] / spacing))):
            image_with_grid[s * spacing:s * spacing + thickness, :, c] = colour[c]
        for s in range(1, int(np.ceil(image.shape[1] / spacing))):
            image_with_grid[:, s * spacing:s * spacing + thickness, c] = colour[c]
    return image_with_grid

def pixelate(image : np.ndarray, pixel_dims : list):
    '''
    Pixelate a gray-scale image by averaging pixel values over tiled areas.
    ---
    Parameters
        image : np.ndarray
            Grey-scale image to be pixelated.
        pixel_dims : list[int, int]
            A list of two integers containing the dimensions of tiled areas over which pixel values
            will be averaged.
    ---
    Outputs
        pixelated_image : np.ndarray
            A pixelated version of the input image.
        reduced_image : np.ndarry
            A version of the pixelated image with the dimensions reduced by a factor equal to the
            specified pixel dimensions.
    '''
    pixelated_image = np.zeros(image.shape, dtype=np.uint8)
    reduced_image = np.zeros((int(np.ceil(image.shape[0] / pixel_dims[0])),
                              int(np.ceil(image.shape[1] / pixel_dims[1]))))###################
    for x_block in range(0, reduced_image.shape[0]):
        limits_x = [x_block * pixel_dims[0],
                    min([(x_block + 1) * pixel_dims[0], image.shape[0]])]
        for y_block in range(0, reduced_image.shape[1]):
            limits_y = [y_block * pixel_dims[1],
                        min([(y_block + 1) * pixel_dims[1], image.shape[1]])]
            pixel = image[limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]]
            pixel_value = int(np.sum(pixel) / (np.prod(pixel_dims)))
            pixelated_image[limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]] = pixel_value
            reduced_image[x_block, y_block] = pixel_value
    return pixelated_image, reduced_image

def find_grid_corners(binary_image : np.ndarray, margin : float, pixel_dims : list):
    '''
    Find the four corner locations of a square which is centered on the middle of the largest
    region of high intensity in the image and is not closer than margin (percent) to the
    surrounding region with 0 intensity.
    ---
    Parameters
        binary_image : np.ndarray
            Image containing a region of high intensity pixels surrounded by pixels with 0
            intensity.
        margin : float
            Value between 0 - 0.5 specifying the relative distance between the square corners and
            the boundary of the region of high intensity.
        pixel_dim : list[int, int]
            Size of pixelation to apply.
    ---
    Outputs
        corners : np.ndarray
            4x2 matrix of corner coordinates.
        reduced_frontier_scaled : np.ndarray
            Inverse intensity map of proximity to black or border pixels.
    '''
    # Start with a lower resolution image for time efficiency.
    pixelated_image, reduced_image = pixelate(binary_image, pixel_dims)
    # Map the pixel values in the reduced image to 1 if on a border of the image, or the pixel...
    # ...intensity is 0, and map to 0 otherwise.
    reduced_frontier = np.where(reduced_image == 0, 1, 0)
    reduced_frontier[[0, -1], :] = 1
    reduced_frontier[:, [0, -1]] = 1
    # Propagate through the frontier to record the hamiltonian distance between each pixel and...
    # ...the nearest pixel in the reduced image that either has 0 intensity or is at a border.
    for k in range(1, min(reduced_image.shape)):
        if np.sum(np.where(reduced_frontier == 0, 1, 0)) == 0:
            break
        neighbours = np.zeros(reduced_frontier.shape)
        neighbours[1:, :] = neighbours[1:, :] + reduced_frontier[:-1, :]
        neighbours[:-1, :] = neighbours[:-1, :] + reduced_frontier[1:, :]
        neighbours[:, :-1] = neighbours[:, :-1] + reduced_frontier[:, 1:]
        neighbours[:, 1:] = neighbours[:, 1:] + reduced_frontier[:, :-1]
        reduced_frontier = np.where((reduced_frontier == 0) & (neighbours > 0),
                                    k + 1, reduced_frontier)
    # Normalise the frontier on to the scale 0 - 255.
    reduced_frontier_scaled = (reduced_frontier - 1) * 255 / np.max(reduced_frontier - 1)
    reduced_rows = np.tile(np.array([range(0, reduced_frontier_scaled.shape[0])]).T,
                           (1, reduced_frontier_scaled.shape[1]))
    reduced_cols = np.tile(np.array([range(0, reduced_frontier_scaled.shape[1])]),
                           (reduced_frontier_scaled.shape[0], 1))
    # Find the furthest pixel from black or border pixels by taking a weighted average.
    reduced_center = np.array([np.sum(reduced_rows * reduced_frontier_scaled) /
                               reduced_frontier_scaled.sum(),
                               np.sum(reduced_cols * reduced_frontier_scaled) /
                               reduced_frontier_scaled.sum()])
    reduced_center_rounded = np.floor(reduced_center).astype(int)
    reduced_diags = np.array([0, 1])
    # Search diagonally from the centroid pixel until specified margin is reached.
    for s in range(0, 2 * min(reduced_image.shape)):
        square = reduced_frontier_scaled[reduced_center_rounded[0] - reduced_diags[0]:
                                         reduced_center_rounded[0] + reduced_diags[1],
                                         reduced_center_rounded[1] - reduced_diags[0]:
                                         reduced_center_rounded[1] + reduced_diags[1]]
        if square.min() < margin * 255:
            break
        else:
            reduced_diags[s % 2] += 1
    # Calculate position of centroid pixel and corners in original image.
    center = np.floor(reduced_center * np.array(pixel_dims)).astype(int)
    side_lengths = (reduced_diags.sum() + 1) * np.array(pixel_dims)
    corners = [[int(center[0] + 0.5 * side_lengths[0]), int(center[1] - 0.5 * side_lengths[1])],
               [int(center[0] + 0.5 * side_lengths[0]), int(center[1] + 0.5 * side_lengths[1])],
               [int(center[0] - 0.5 * side_lengths[0]), int(center[1] + 0.5 * side_lengths[1])],
               [int(center[0] - 0.5 * side_lengths[0]), int(center[1] - 0.5 * side_lengths[1])]]
    return corners, reduced_frontier_scaled.astype(np.uint8)

def measure_intensities(image : np.ndarray, points : np.ndarray, scan_range : int):
    '''
    Extracts the total, summed intensities of pixels around a set of points.
    ---
    Parameters
        image : np.ndarray
            Picture from which pixel intensities will be measured.
        points : np.ndarray
            Nx2 array containing N coordinates around which pixel intensities will be summed.
        scan_range : int
            Maximum distance in each axis of measured pixels from the specified coordinates.
    ---
    Outputs
        total_intensities : np.ndarray
            Nx0 array of total measured pixel intensities for each specified coordinate.
    '''
    total_intensities = np.zeros(points.shape[0])
    for p in range(0, points.shape[0]):
        square = image[points[p][0] - scan_range:points[p][0] + scan_range,
                       points[p][1] - scan_range:points[p][1] + scan_range]
        total_intensities[p] = square.sum()
    return total_intensities

def get_bright_lines(intensities : list):
    '''
    Identifies indexes with globally maximal recorded intensities and returns the approximate
    locations of the sampling regions.
    ---
    Parameters
        intensities : list[np.ndarray, np.ndarray]
            List containing two arrays of shapes NxA / NxB respectively, where N is the number
            of sampled regions and A / B is the number of scanning rows / columns.
    ---
    Outputs
        bright_lines : 
    '''
#     bright_lines= [[[] for _ in range(0, intensities[d].shape[0])]
#                    for d in range(0, len(intensities))]
    bright_lines = -1 * np.ones((intensities[0].shape[0], 2))
#     envelope = np.array([-1, 0, 1])
    for d in range(0, len(intensities)):
        #sorted_intensities = np.sort(intensities[d], 1)
        for c in range(0, intensities[d].shape[0]):
            dir_corner_ints = intensities[d][c, :]
            envelope_totals = dir_corner_ints[:-2] + dir_corner_ints[1:-1] + dir_corner_ints[2:]
            main_line = np.argmax(envelope_totals)
            bright_lines[c, d] = (((main_line - 1) * dir_corner_ints[main_line - 1] +
                                   main_line * dir_corner_ints[main_line] +
                                   (main_line + 1) * dir_corner_ints[main_line + 1]) /
                                  np.sum(dir_corner_ints[main_line - 1:main_line + 2]))
#             #set threshold as twice the median intensity value, add one to set above 0
#             threshold = 2 * np.sort(dir_corner_ints)[int(len(dir_corner_ints) / 2)] + 1#########
#             num_checks = np.sum((dir_corner_ints >= threshold).astype(int))############
#             lines_checked = []
#             for _ in range(0, num_checks):
#                 max_index = np.argsort(dir_corner_ints)[-1]###########
#                 if len(set.intersection(set(max_index + envelope),
#                                         set(lines_checked))) == 0:
#                     bright_lines[d][c].append(max_index)
#                 lines_checked.append(max_index)
#                 dir_corner_ints[max_index] = 0
#                 # For simplicity, only output the single brightest line
#                 break
    return bright_lines

def validate_calibration(camera2projector : [np.ndarray, str], size=40, duration=5):   
    if isinstance(camera2projector, str):
        camera2projector = np.load(camera2projector)
        
    dome_pi4node.transmit({"screen": 'new'})
    response = dome_pi4node.receive()
    dome_pi4node.transmit({"scale": 1})
    response = dome_pi4node.receive()
    dome_pi4node.transmit(np.array([0,0,0], dtype=np.uint8))
    response = dome_pi4node.receive()
    
    pos_cam =[]
    pos_proj =[]
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,0)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,0)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,1920)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,1920)))
    
    for pos in pos_cam:
        pos_proj.append(np.dot(camera2projector, pos))
    
    cmd = { "add0": {"label": 'a', "shape type": 'square',"pose": pos_proj[0], "colour": [0, 0, 100]},
            "add1": {"label": 'a', "shape type": 'square',"pose": pos_proj[1], "colour": [0, 100, 100]},
            "add2": {"label": 'a', "shape type": 'square',"pose": pos_proj[2], "colour": [100, 0, 100]},
            "add3": {"label": 'a', "shape type": 'square',"pose": pos_proj[3], "colour": [0, 100, 0]}}
    dome_pi4node.transmit(cmd)
    out_msg = dome_pi4node.receive()
    
    time.sleep(duration)
    return out_msg


def start_calibration(out_file : str, sq_size = 10):
    projector_dims = (480, 854, 3)
    camera_mode = 'default'
    response = None
    
    if os.path.isfile(out_file):
        camera2projector = np.load(out_file)
    else:
        camera2projector = DOMEtran.linear_transform(scale=0.25)
    
    dome_pi4node.transmit({"scale": 1})
    response = dome_pi4node.receive()
    
    # project three squares
    delta = 50
    off_color = [150, 150, 150]
    on_color  = [150, 255, 150]
    
    pos =[]
    pos.append( DOMEtran.linear_transform(scale=sq_size, shift=(0,0)) )
    pos.append( DOMEtran.linear_transform(scale=sq_size, shift=(0,1920)) )
    pos.append( DOMEtran.linear_transform(scale=sq_size, shift=(1080,0)) )
    
    for i in range(len(pos)):
        pos[i] = np.dot(camera2projector, pos[i])
    
    cmd = {"screen": '0',
            "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": off_color},
            "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": off_color},
            "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": off_color}}
    dome_pi4node.transmit(cmd)
    response = dome_pi4node.receive()       
    
    print(f'Welcome to the DOME calibration set up procedure.\nAt any point in the ' \
    f'calibration, if all requisite parameters have been specified, input "skip" ' \
    f'to begin scanning. Alternatively, enter "next" to proceed to the next step, ' \
    f'"back" to return to the previous step, "restart" to begin the calibration ' \
    'from the start, or "exit" to end the program.')
        
    threshold_picture = None
    loaded_file = None
    saved_file = None
    step = 1
    count = 1
    while 1 <= step <= 10:

        # STEP 1: adjust the focus
        while step == 1:
            message = f'--- STEP 1 ---\nTo begin, we will focus the camera by adjusting ' \
                      f'the height of the sample stage on the DOME. Input the number of ' \
                      f'seconds for which to display a live feed of the camera frame as ' \
                      f'an integer. By turning the lead screw on the DOME, move the ' \
                      f'sample stage up or down until the image comes in to focus. Once ' \
                      f'this is done, input "next" to continue. Henceforth, it is ' \
                      f'crucial that the camera, projector, sample stage and any lenses ' \
                      f'maintain their relative positions. Any change to the physical ' \
                      f'setup of the DOME may invalidate the calibration.\n'
            
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    duration = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            time.sleep(1)
        if step != 0:
            loaded_file = None
            
        # STEP 2: first square y pos    
        while step == 2:
            s = 0
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": on_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": off_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": off_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 2 ---\nMove the first square to the top of the camera view.'\
                      f' To move the square input positive or negative values. \n'\
                      f' Enter next when you are done. Enter back to go to the previous step.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][0,2]+=increment

        # STEP 3: first square x pos     
        while step == 3:
            s = 0
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": on_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": off_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": off_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 3 ---\nMove the first square to the left side of the camera view.'\
                      f' To move the square input positive or negative values.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][1,2]+=increment
        
        # STEP 4: second square y pos    
        while step == 4:
            s = 1
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": off_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": on_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": off_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 4 ---\nMove the second square to the top of the camera view.'\
                      f' To move the square input positive or negative values.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][0,2]+=increment

        # STEP 5: second square x pos     
        while step == 5:
            s = 1
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": off_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": on_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": off_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 5 ---\nMove the second square to the right side of the camera view.'\
                      f' To move the square input positive or negative values.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][1,2]+=increment
       
        # STEP 6: third square y pos    
        while step == 6:
            s = 2
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": off_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": off_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": on_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 6 ---\nMove the third square to the bottom of the camera view.'\
                      f' To move the square input positive or negative values.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][0,2]+=increment

        # STEP 7: third square x pos     
        while step == 7:
            s = 2
            cmd = {"screen": f'{count}',
                "add1": {"label": 'a', "shape type": 'square',"pose": pos[0].tolist(), "colour": off_color},
                "add2": {"label": 'a', "shape type": 'square',"pose": pos[1].tolist(), "colour": off_color},
                "add3": {"label": 'a', "shape type": 'square',"pose": pos[2].tolist(), "colour": on_color}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            count+=1
            
            message = f'--- STEP 7 ---\nMove the third square to the left side of the camera view.'\
                      f' To move the square input positive or negative values.\n'
            user_args, step = custom_input(message, step)
            if len(user_args) == 1 and len(user_args[0]) != 0:
                try:
                    increment = int(user_args[0])
                except ValueError:
                    print('Please input an integer.')
                    continue
            elif step != 1:
                continue
            #move = DOMEtran.linear_transform(shift=(step,0))
            #cmd = {'transform': {'matrix': move.tolist(), 'labels': ['sq1']}}
            pos[s][1,2]+=increment
      
        # STEP 8: perform calibration
        if step == 8:
            camera_points = np.float32([[0,0], [0,1920], [1080,0]])
            projector_points = np.float32([[0,0]]*3)
            projector_points[0] = pos[0][0:2,2]
            projector_points[1] = pos[1][0:2,2]
            projector_points[2] = pos[2][0:2,2]
            camera2projector = cv2.getAffineTransform(camera_points,
                                                      projector_points)
            camera2projector = np.concatenate((camera2projector, np.array([[0, 0, 1]])), 0)
            print('camera2projector='); print(camera2projector)
            step += 1
            
        # STEP 9: validation
        if step == 9:
            print('The camera frame should now be exactly filled by a green frame.')
            green_cam = DOMEtran.linear_transform(scale=(1080,1920), shift=(1080/2,1920/2))
            black_cam = DOMEtran.linear_transform(scale=(1080-sq_size,1920-sq_size), shift=(1080/2,1920/2))
            green_proj = np.dot(camera2projector, green_cam)
            black_proj = np.dot(camera2projector, black_cam)
            cmd = {"screen": 'new',
                   "add1": {"label": 'a', "shape type": 'square',"pose": green_proj.tolist(), "colour": on_color},
                   "add2": {"label": 'a', "shape type": 'square',"pose": black_proj.tolist(), "colour": [0, 0, 0]}}
            dome_pi4node.transmit(cmd)
            response = dome_pi4node.receive()
            user_args, step = custom_input('Input \'next\' to finish and save the calibration or \'back\' to adjust it.\n', step)
         
        # STEP 10: save the transformation matrix in 'camera2projector.npy'
        if step == 10:
            np.save(out_file, camera2projector)
            print(f'Calibration complete.\n--- Affine transform saved to {out_file}')
            dome_pi4node.transmit('all' + 3 * ' 0')
            response = dome_pi4node.receive()
            step += 1
            
    dome_pi4node.transmit('all' + 3 * ' 0')
    response = dome_pi4node.receive()


if __name__ == '__main__':
    out_folder = '/home/pi/Documents/config'
    name = 'camera2projector_x90'
    sq_size = 40
    
    date = datetime.today().strftime('%Y_%m_%d')
    name_date = name + '_' + date + '.npy'
    calibration_file = os.path.join(out_folder,name_date)
    
    dome_pi4node = DOMEcomm.NetworkNode()
    dome_camera = DOMEutil.CameraManager()
    
    # connect to projector
    print('On the projector Pi run "DOME_calibration_projector.py" and wait for a black ' \
          'screen to appear (this may take several seconds). Once a black screen is ' \
          'shown, click anywhere on the black screen, then press any key (such as ALT).')
    dome_pi4node.accept_connection()

    # start live video from the camera
    dome_camera.camera.start_preview()
    dome_camera.camera.preview.fullscreen=False
    dome_camera.camera.preview.window=(1000, 40, 854, 480)
    
    print('Use start_calibration(calibration_file, sq_size) to start the calibration.\n'\
          'If out_file already exists it will be loaded and modified.\n'\
          'Use validate_calibration(calibration_file, sq_size) to test an existing calibration.')    
    
    