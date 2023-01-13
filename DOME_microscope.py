# This code is provided to easly control the DOME using the console. The
# DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio Denniss. This code requires
# the "DOME_caibration_projector.py" file to be run in parallel on the Raspberry Pi 0 connected to
# the DOME projector.
# #################################################################################################
# Authors = Andrea Giusti <andrea.giusti@unina.it>
# Affiliation = University of Naples Federico II
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import DOME_communication as DOMEcomm
import DOME_imaging_utilities as DOMEutil
import DOME_experiment_manager as DOMEexp

import numpy as np
import cv2
import json
import time
import os
import signal
import atexit
from pathlib import Path
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


def load_calibration_settings(file_name : str, keys : list):
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


def init(camera_settings=None, gpio_light=None):
    '''
    Initialize communication with the projector and camera modules.
    ---
    Outputs:
        dome_pi4node : NetworkNode
            Network node associated to the imaging module.
        dome_camera : CameraManager
            Camera manager to control the camera.
        dome_gpio : PinManager
            GPIO manager to control the pins of the board.
    '''
    global output_directory
    camera_mode = 'default'
    response = None

    dome_pi4node = DOMEcomm.NetworkNode()
    save_dir = os.path.join(output_directory,'default')
    dome_camera = DOMEutil.CameraManager(save_dir=save_dir)
    dome_gpio = DOMEutil.PinManager()
    
    try:
        if not camera_settings is None:
            camera_mode = 'custom'
            dome_camera.store_settings(camera_mode, camera_settings)
        if not gpio_light is None:
            dome_gpio.add_pin_label('light source', gpio_light)
            dome_gpio.toggle('light source', 'on')
        print('On the projector Pi run "DOME_calibration_projector.py" and wait for a black ' \
              'screen to appear (this may take several seconds). Once a black screen is ' \
              'shown, click anywhere on the black screen, then press any key (such as ALT).')

        dome_pi4node.accept_connection()
        
        print('\nNow use the provided functions to control the camera and the projector.\n')
        
    except:
        print('Connection error')
        
    return dome_pi4node, dome_camera, dome_gpio

def set_color(newcolor):
    '''
    Set projection color.
    ---
    Parameters
        newcolor : str or list
            New color string (white, red, green and blue) or three elements list.
    '''
    global color
    
    if type(newcolor) is str and newcolor in ['white', 'red', 'green', 'blue']:
        
        if newcolor=='white':
            color = np.array([1, 1, 1])
        elif newcolor=='blue':
            color = np.array([1, 0, 0])
        elif newcolor=='red':
            color = np.array([0, 0, 1])
        elif newcolor=='green':
            color = np.array([0, 1, 0])
        else:
            print('Non valid color string.\n')
        
        update_projector()
        print('Color set.\n')
        if current_experiment:
            current_experiment.add_log_entry(f'color={newcolor}')
        
    elif type(newcolor) is list and len(newcolor)==3:
        if any(newcolor)>1 or any(newcolor)<0:
            raise(Exception('Use values between 0 and 1.\n'))
        
        color = np.array(newcolor)
        update_projector()
        print('Color set.\n')
        if current_experiment:
            current_experiment.add_log_entry(f'color={newcolor}')
    else:
        print('Input a string white, red, green and blue, or a list of three elements between 0 and 1.\n')
        
def set_brightness(newbright : int):
    '''
    Set projection brightness.
    ---
    Parameters
        newbright : int
            Brightness value (0-250).
    '''
    if newbright>=0 and newbright<=255:
        global bright
        bright = newbright
        response=update_projector()
        
        if response != 'Done':
            raise(Exception('Error communicating with the projector.\n'))
        
        print(f'Brightness set to {bright}.\n')
        
        if current_experiment:
            current_experiment.add_log_entry(f'bright={bright}')
        
    else:
        print('Input a value between 0 and 255.\n')
    
def update_projector():
    '''
    Send command to the projector module.
    ---
    Outputs
        response : int
            Response from the projector module.
    '''
    light = bright * color
    dome_pi4node.transmit(f'all' + f' {light[0]} {light[1]} {light[2]}')
        
    response = dome_pi4node.receive()
    return response

def capture(file_name = '', autosave=True, show=False):
    '''
    Capture still image from the camera.
    ---
    Parameters
        file_name = ''
            Name of the file to save the image (extension escluded).
            If not provided an automatic name is generated.
        autosave=True
            Save the acquired image using the given file name.
        show=False
            Open a window showing the acquired image.
    ---
    Outputs:
        image : np.ndarray
    '''
        
    image=dome_camera.capture_image()
    
    if current_experiment:
        current_experiment.add_log_entry('image captured: '+file_name)

    if show:
        cv2.imshow('Captured image', image)
        cv2.waitKey(0)
    if autosave:
        print('Image captured.\n')
        save(image, file_name)
    else:
        print('Image captured. Use save(image, file_name) to store it.\n')
    
    return image

def save(image : np.ndarray, file_name = ''):
    '''
    Save image acquired with capture() command.
    ---
    Parameters
        image : np.ndarray
            Image acquired with capture() command.
        file_name = ''
            Name of the file to save the image (extension escluded).
            If not provided an automatic name is generated.
    '''
        
    if file_name == '':
        global fig_counter
        image_file_name = f'fig{fig_counter}.jpeg'
        fig_counter=fig_counter+1
    else:
        image_file_name = file_name + '.jpeg'
        
    try:
        dome_camera.save_image(image_file_name, image)
        print('Figure saved on file: ' + image_file_name + '\n')
    except:
        print('First use capture to get an image')


def rec(file_name = ''):
    '''
    Acquire and save video from the camera.
    ---
    Parameters
        file_name = ''
            Name of the file to save the video (extension escluded).
            If not provided an automatic name is generated.
    '''
    if file_name == '':
        global video_counter
        video_file_name = f'video{video_counter}.h264'
        video_counter=video_counter+1
    else:
        video_file_name = file_name + '.h264'
    
    dome_camera.rec(video_file_name)
    
    if current_experiment:
        current_experiment.add_log_entry(f'video recording started: {video_file_name}')
    
    print('Recording started on file: ' + video_file_name + '\nUse stop_rec() to stop the recording.\n')

def stop_rec():
    '''
    Stop video recording.
    '''
    dome_camera.stop_rec()
    
    if current_experiment:
        current_experiment.add_log_entry(f'video recording stopped')
    
    print('Recording stopped.\n')
    
def close_camera():
    '''
    Stop live video from the camera.
    '''
    dome_camera.camera.stop_preview()
    print('Live view stopped. Use open_camera() to restart.\n')
    
def open_camera( window=(1000, 40, 1000, 400) ):
    '''
    Start live video from the camera.
    ---
    Parameters
        window = (1000, 40, 1000, 400)
            Position and dimensions of the preview window.
    '''
    dome_camera.camera.start_preview()
    dome_camera.camera.preview.fullscreen=False
    dome_camera.camera.preview.window=window
    print('Live view started.\n')

def store_camera_settings(mode_name : str, camera_settings : dict):
    '''
    Store camera settings.
    ---
    Parameters
        mode_name : str
            Name of the store the settings.
        camera_settings : dict
            Settings to store.
    '''
    dome_camera.store_settings(mode_name, camera_settings)
    print('Camera settings stored.\n')
    
def load_camera_settings(mode_name : str):
    '''
    Apply previously stored settings to the camera.
    ---
    Parameters
        mode_name : str
            Name of the settings to be loaded to the camera.
    '''
    dome_camera.load_settings(mode_name)
    
    if current_experiment:
        current_experiment.add_log_entry(f'camera settings loaded: {mode_name}')
    
    print('Camera settings loaded.\n')
    
def set_camera_value(setting_name : str, value, mode_name='', autoload=True):
    '''
    Set the value of a setting of a camera mode.
    ---
    Parameters 
        setting_name : str
            Name of the setting to modify (iso, framerate, etc...).
        value
            New value of the setting. Must be of a proper type.
        mode_name : str
            Name of the camera mode to modify.
    '''
    
    dome_camera.set_value(setting_name, value, mode_name, autoload)
    
    if current_experiment and autoload:
        current_experiment.add_log_entry(f'camera {setting_name}={value}')
            
def disconnect():
    '''
    Disconnect projector module.
    '''
    dome_pi4node.transmit('all' + 3 * ' 0')
    response = dome_pi4node.receive()
    dome_pi4node.transmit('exit')
    response = dome_pi4node.receive()
    print('Projector disconnected.\n')

def new_experiment(date : str, species : str, culture ='', output_directory='/home/pi/Documents/experiments'):
    global current_experiment
    global dome_camera
    
    current_experiment= DOMEexp.ExperimentManager(date, species, culture, output_directory)    
    dome_camera.save_dir = current_experiment.path
    print(f'Now working in new folder {current_experiment.path}')
    
def load_experiment(name : str, output_directory='/home/pi/Documents/experiments'):
    global current_experiment
    global dome_camera
    
    current_experiment= DOMEexp.open_experiment(name, output_directory)    
    dome_camera.save_dir = current_experiment.path
    print(f'Now working in folder {current_experiment.path}')

if __name__ == '__main__':
    
    output_directory='/home/pi/Documents/experiments'
    current_experiment=None
    
    #color = 'white' # projected light color: white, red, blue or green 
    color=np.array([1, 1, 1])
    bright = 10     # brightness 0-255
    
    fig_counter = 1
    video_counter = 1
    
    [dome_pi4node, dome_camera, dome_gpio]=init()
    
    # start video preview
    set_camera_value('brightness', 40)
    open_camera()
    dome_camera.show_info()
    
    # initialize projector
    update_projector()
    
    # handle program termination
    atexit.register(disconnect)
    signal.signal(signal.SIGTERM, disconnect)
    signal.signal(signal.SIGINT, disconnect)
    signal.signal(signal.SIGABRT, disconnect)
    signal.signal(signal.SIGILL, disconnect)
    signal.signal(signal.SIGSEGV, disconnect)
