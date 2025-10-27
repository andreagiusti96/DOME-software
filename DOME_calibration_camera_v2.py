#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOME_calibration_camera_v2.py
This code is provided to guide users through the process of calibration to find a mathematical
transformation that describes the mapping of camera pixels to projector pixels in the DOME. 

This code must run on the Raspberry Pi connected to the DOME camera.
This code requires the "DOME_caibration_projector.py" file to be run in parallel on the 
Raspberry Pi 0 connected to the DOME projector. 

To calibrate the DOME, run this file and follow the on screen instructions.

See also: DOME_calibration_camera.

Authors:    Andrea Giusti and Matthew Uppington
Created:    2023
"""

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


def start_calibration(out_file : str, sq_size = 40):
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
    
    cmd = {"screen": 'new',
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
    while 1 <= step <= 9:

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
            pos[s][1,2]+=increment
      
        # STEP 8: perform calibration and validation
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
         
        # STEP 9: save the transformation matrix in 'camera2projector.npy'
        if step == 9:
            np.save(out_file, camera2projector)
            print(f'Calibration complete.\n--- Affine transform saved to {out_file}')
            dome_pi4node.transmit('all' + 3 * ' 0')
            response = dome_pi4node.receive()
            step += 1
            
    dome_pi4node.transmit('all' + 3 * ' 0')
    response = dome_pi4node.receive()


if __name__ == '__main__':
    out_folder = '/home/pi/Documents/config'
    name = 'camera2projector_x36'
    sq_size = 40
    
    date = datetime.today().strftime('%Y_%m_%d')
    name_date = name + '_' + date + '.npy'
    calibration_file = os.path.join(out_folder,name_date)
    
    dome_pi4node = DOMEcomm.NetworkNode()
    dome_camera = DOMEutil.CameraManager()
    
    # connect to projector
    print('On the projector Pi run "DOME_projection_interface.py" and wait for a black ' \
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
    
    