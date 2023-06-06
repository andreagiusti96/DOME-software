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
import DOME_transformation as DOMEtran
import DOME_projection_interface as DOMEproj

import numpy as np
import cv2
import json
import time
import os
import signal
import atexit
from datetime import datetime
from datetime import date
from scipy import signal as sig


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
        print('On the projector Pi run "DOME_projection_interface.py" and wait for a black ' \
              'screen to appear (this may take several seconds). Once a black screen is ' \
              'shown, click anywhere on the black screen, then press any key (such as ALT).')

        dome_pi4node.accept_connection()
        
        print('\nNow use the provided functions to control the camera and the projector.\n')
        
    except:
        print('Connection error')
        
    return dome_pi4node, dome_camera, dome_gpio

def save_to_json(var, file_name : str):
    if '.json' not in file_name:
        file_name = file_name + '.json'
    DOMEcomm.recursive_tolist(var)
    with open(file_name, 'w') as file:
        json.dump(var, file, indent=2)
    
def load_json(file_name : str):
    '''
    Reads calibration setting values from a json file.
    '''
    if '.json' not in file_name:
        file_name = file_name + '.json'
    with open(file_name, 'r') as file:
        var = json.load(file)
    return var

def set_color(newcolor, prevent_print=False, prevent_log=False):
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
        
        update_projector(prevent_log=prevent_log)
        #print('Color set.\n')
#         if current_experiment and not prevent_log:
#             current_experiment.add_log_entry(f'color={newcolor}')
        
    elif type(newcolor) in [list, type(np.array([1]))] and len(newcolor)==3:
        if any(newcolor)>1 or any(newcolor)<0:
            raise(Exception('Use values between 0 and 1.\n'))
        
        color = np.array(newcolor)
        update_projector(prevent_log=prevent_log)
        #print('Color set.\n')
#         if current_experiment and not prevent_log:
#             current_experiment.add_log_entry(f'color={newcolor}')
    else:
        print('Input a string white, red, green and blue, or a list of three elements between 0 and 1.\n')
 
def set_brightness(newbright : int, prevent_log=False):
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
        update_projector(prevent_log=prevent_log)
        
        print(f'Brightness set to {bright}.\n')
        
#         if current_experiment and not prevent_log:
#             current_experiment.add_log_entry(f'bright={bright}')
        
    else:
        print('Input a value between 0 and 255.\n')
     
# def make_pattern(dimensions):
#     '''
#     Generate pattern for the projector.
#     ---
#     Outputs
#         pattern : np.ndarray([480, 854, 3], dtype=np.uint8)
#             Generated patter.
#     '''
#     
#     light = np.rint( bright * color )
#     pattern = np.ndarray(dimensions, dtype=np.uint8)
#     pattern[:, :, 0] = int(light[0])
#     pattern[:, :, 1] = int(light[1])
#     pattern[:, :, 2] = int(light[2])
#     
#     return pattern

def update_pattern(command, prevent_log=False):
    global screen_manager
    update_projector(command, prevent_log)
    pattern, msg_out = screen_manager.make_pattern_from_cmd(command)
    return pattern, msg_out

def update_projector(command=None, prevent_log=False):
    '''
    Send command to the projector module.
    ---
    Parameters
        command=None
            Command to send to the projector.
            If None a unifor pattern is generated with bright and color.
    ---
    Outputs
        response : str
            Response from the projector module.
    '''
    if command is None: 
        light = np.rint( bright * color )
        dome_pi4node.transmit(f'all' + f' {int(light[0])} {int(light[1])} {int(light[2])}')
        if current_experiment and not prevent_log:
            current_experiment.add_log_entry(f'light={light}')
    else:
        if isinstance(command, np.ndarray):
            command.astype(np.uint8)
        
        dome_pi4node.transmit(command)
        if current_experiment and not prevent_log:
            current_experiment.add_log_entry(f'projector updated')

    response = dome_pi4node.receive()
    
    return response

def capture(file_name = '', autosave=True, show=False, prevent_print=False, prevent_log=False):
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
    
    if current_experiment and not prevent_log:
        current_experiment.add_log_entry('image captured: '+file_name)

    if show:
        cv2.imshow('Captured image', image)
        cv2.waitKey(0)
    if autosave:
        if not prevent_print: print('Image captured.\n')
        save(image, file_name, prevent_print)
    else:
        print('Image captured. Use save(image, file_name) to store it.\n')
    
    return image

def save(image : np.ndarray, file_name = '', prevent_print=False):
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
        if not prevent_print:
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
    
def open_camera( window=(1000, 40, 854, 480) ):
    '''
    Start live video from the camera.
    ---
    Parameters
        window = (1000, 40, 854, 480)
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
    
def load_camera_settings(mode_name=''):
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

def new_experiment(date : str, species : str, culture ='', output_directory='/home/pi/Documents/experiments'):
    global current_experiment
    global dome_camera
    
    current_experiment= DOMEexp.ExperimentManager(date, species, culture, output_directory)    
    dome_camera.save_dir = current_experiment.path
    print(f'Now working in new folder {current_experiment.path}')
    return current_experiment
    
def load_experiment(name : str, output_directory='/home/pi/Documents/experiments'):
    global current_experiment
    global dome_camera
    
    current_experiment= DOMEexp.open_experiment(name, output_directory)    
    dome_camera.save_dir = current_experiment.path
    print(f'Now working in folder {current_experiment.path}\n')
    return current_experiment

def get_index_for_time(time : float):
    index=int(time/deltaT)
    return index

def validate_calibration(camera2projector : np.ndarray, size=50):
    # get prevous pattern scale and set to 1
    old_scale = update_projector({'get':{'param':'scale'}})
    update_pattern({"scale": 1}, prevent_log=True)
    
    pos_cam =[]
    pos_proj =[]
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,0)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,0)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,1920)))
    pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,1920)))
    
    for pos in pos_cam:
        pos_proj.append(np.dot(camera2projector, pos))
    
    cmd = {"screen": 'new',
            "add0": {"label": 'a', "shape type": 'square',"pose": pos_proj[0], "colour": [0, 0, 100]},
            "add1": {"label": 'a', "shape type": 'square',"pose": pos_proj[1], "colour": [0, 100, 100]},
            "add2": {"label": 'a', "shape type": 'square',"pose": pos_proj[2], "colour": [100, 0, 100]},
            "add3": {"label": 'a', "shape type": 'square',"pose": pos_proj[3], "colour": [0, 100, 0]}}
    out_msg = update_projector(cmd)
    
    # reset previous pattern scale
    time.sleep(2)
    update_pattern({"scale": old_scale}, prevent_log=True)
    return out_msg

def start_experiment():
    # start the experiment
    global current_experiment
    global screen_manager

    current_experiment=new_experiment(date, species, culture, output_directory)
    current_experiment.add_detail(f'Sample: '+sample, include_in_exp_list=True)
    current_experiment.add_detail(f'Duration={totalT}s', include_in_exp_list=True)
    current_experiment.add_detail(f'Sampling time={deltaT}s\n')
    current_experiment.add_detail(f'Sample temperature={temp}°C\n')
    current_experiment.add_detail(f'Calibration file = {camera2projector_file}')
    current_experiment.add_detail('Camera settings:\n'+dome_camera.print_settings()+'\n')
    current_experiment.add_detail(f'camera_bright_reduction={camera_bright_reduction}\n')
    current_experiment.add_detail('Log:')
    os.mkdir(os.path.join(current_experiment.path, 'patterns'))
    os.mkdir(os.path.join(current_experiment.path, 'patterns_cam'))
    os.mkdir(os.path.join(current_experiment.path, 'images'))
    
    count=0
    cmd_count=0
    
    update_pattern({"screen": 'new'}, prevent_log=True)
    pattern, msg_out = update_pattern(off_light, prevent_log=True)
    
    current_experiment.reset_starting_time()
    rec('video')
    print('Experiment running...\n')
    
    # run the experiment
    for count in range(0,max_time_index):

        # acquire data
        t=count*deltaT
        tic=datetime.now()
        activation_times[count]=(tic - current_experiment.start_time).total_seconds()
        out_img=os.path.join('images', 'fig_' + '%04.1f' % t)
        #images[count,:,:,:]=capture(out_img, prevent_print=True, prevent_log=False)
        capture(out_img, prevent_print=True, prevent_log=False)
        
#         # compute output
#         output=outputs[count]
#         newcolor=off_value*(1-output)+on_value*output
#         #dome_camera.camera.brightness=int(camera_bright_base-camera_bright_reduction*output)
        
        # apply output
        if cmd_count < len(commands):
            if t >= commands[cmd_count]["t"]:
                message = commands[cmd_count]["cmd"]
                #update_projector(message, prevent_log=False)
                #pattern, msg_out = screen_manager.make_pattern_from_cmd(message)
                pattern, msg_out = update_pattern(message)
                pattern_cam = DOMEtran.transform_image(pattern, projector2camera, camera_dim)
                cmd_count+=1
        
        # save output pattern
        out_patt=os.path.join(current_experiment.path, 'patterns', 'pattern_' + '%04.1f' % t + '.jpeg')
        cv2.imwrite(out_patt, pattern)
        out_patt=os.path.join(current_experiment.path, 'patterns_cam', 'pattern_' + '%04.1f' % t + '.jpeg')
        cv2.imwrite(out_patt, pattern_cam)
        
        # wait
        toc=datetime.now()
        ellapsed_time=toc - current_experiment.start_time
        time_to_wait=t+deltaT-ellapsed_time.total_seconds()
        if time_to_wait<-0.01:
            print(f't={t}: {-time_to_wait:3.2}s delay!\n')
        else:
            time.sleep(max(0, time_to_wait ))
    
    # terminate the experiment and recording
    update_pattern({"screen": 'new'}, prevent_log=True)
    pattern, msg_out = update_pattern(off_light, prevent_log=True)
    terminate_experiment()

def terminate_experiment():
    global current_experiment
    
    print('Experiment stopped.\n')
    stop_rec()
    print('Saving data...\n')
    #current_experiment.save_data(title="data", activation_times=activation_times, images=images, patterns=patterns)
    current_experiment.save_data(title="data",
                                 activation_times=activation_times,
                                 commands = commands,
                                 camera2projector=camera2projector,
                                 scale=scale)
    current_experiment=None

def disconnect():
    '''
    Disconnect projector module.
    '''
    dome_pi4node.transmit('all' + 3 * ' 0')
    response = dome_pi4node.receive()
    dome_pi4node.transmit('exit')
    response = dome_pi4node.receive()
    print('Projector disconnected.\n')

def terminate_session():
    if current_experiment:
        terminate_experiment()
    
    disconnect()
    print('Session tereminated.\n')
    exit()

if __name__ == '__main__':
    
    # details of the experiment
    date='today'    # date of the experiment. Use format YYYY_MM_DD
    species='Prova'     # species used in the experiment
    culture='XXXX'     # culture used in the experiment
    sample='XX'      # details about the sample (volume, frame, etc)
    temp='XX' # temperature of the sample
    
    output_directory      = '/home/pi/Documents/experiments'
    commands_file         = '/home/pi/Documents/config/cmd_prova.json'
    camera2projector_file = '/home/pi/Documents/config/camera2projector_x9_2023_06_06.npy'
    
    deltaT= 0.5 # sampling time [s]
    totalT= 15  # experiment duration [s]
    
    scale = 5
    bright= 200
    
    white = np.array([1, 1, 1])
    black = np.array([0, 0, 0])
    blue  = np.array([1, 0, 0])
    red   = np.array([0, 0, 1])
    green = np.array([0, 1, 0])

    off_value = red*0.1
    on_value = blue + off_value
    off_light = np.rint( bright * off_value ).astype(np.uint8)
    on_light = np.rint( bright * on_value ).astype(np.uint8)
    
    camera_bright_base=40
    camera_bright_reduction=0
    
    # allocate vars
    current_experiment=None
    fig_counter = 1
    video_counter = 1
    
    max_time_index = int(totalT/deltaT + 1)
    activation_times=np.ndarray([max_time_index])
    #images = np.ndarray([max_time_index, 1080, 1920, 3], dtype=np.uint8)
    #patterns = np.ndarray([max_time_index, 480, 854, 3], dtype=np.uint8)
    
    camera_dim = (1080, 1920, 3)
    proj_dim   = ( 480,  854, 3)
    
    camera2projector = np.load(camera2projector_file)
    projector2camera = np.linalg.inv(camera2projector)

    camera_frame = np.array(DOMEtran.map_coordinates(
                    np.array([0,camera_dim[0]]), np.array([0,camera_dim[1]]),
                    camera2projector)) 
    camera_center = np.mean(camera_frame, 1)
    camera_scale = np.diff(camera_frame)
    
    # experiment description
    # load saved commands from a json file or define new ones
    # commands = load_json(commands_file)    

    # commands at specific times
    camera_pose = DOMEtran.linear_transform(scale=camera_scale, shift=camera_center)
    circ_pose = DOMEtran.linear_transform(scale=np.min(camera_scale)/4, shift=camera_center)
    green_cam = camera_pose @ DOMEtran.linear_transform(scale=1)
    black_cam = camera_pose @ DOMEtran.linear_transform(scale=0.9)
    
#     pos_cam =[]
#     pos_proj =[]
#     pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,0)))
#     pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,0)))
#     pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(0,1920)))
#     pos_cam.append( DOMEtran.linear_transform(scale=size, shift=(1080,1920)))
#    
#     for pos in pos_cam:
#         pos_proj.append(np.dot(camera2projector, pos))
    
    commands = [{"t":0, "cmd": on_light},
                {"t":2, "cmd": off_light},
                {"t":3, "cmd": {"add": {"label": 'prova', "shape type": 'circle',
                                "pose": circ_pose, "colour": [0, 50, 0]}}},
                {"t":4, "cmd": {"screen": 'new',
                                "add": {"label": 'prova', "shape type": 'square',
                                "pose": green_cam, "colour": [0, 50, 0]}}},
                {"t":6, "cmd": {"add": {"label": 'prova', "shape type": 'square',
                                "pose": black_cam, "colour": [0, 0, 0]}}},
                {"t":8, "cmd": {"screen": 'new',
                                "add": {"label": 'prova', "shape type": 'circle',
                                "pose": circ_pose, "colour": [0, 50, 0]}}},
                {"t":10, "cmd": {"gradient": {'points': camera_frame[1,:], 'values': [off_light, off_light*10]}}}
                ]
    
#     # varying frequency
#     outputs[get_index_for_time(10):get_index_for_time(15)]=sig.square(2*np.pi*(time_instants[get_index_for_time(10):get_index_for_time(15)]+deltaT/2))*0.5+0.5
#     outputs[get_index_for_time(15):get_index_for_time(20)]=sig.square(0.5*2*np.pi*(time_instants[get_index_for_time(15):get_index_for_time(20)]+deltaT/2))*0.5+0.5
#     outputs[get_index_for_time(20):get_index_for_time(40)]=sig.square(0.2*2*np.pi*(time_instants[get_index_for_time(20):get_index_for_time(40)]+deltaT/2))*0.5+0.5
#     outputs[get_index_for_time(40):get_index_for_time(60)]=sig.square(0.1*2*np.pi*(time_instants[get_index_for_time(40):get_index_for_time(60)]+deltaT/2))*0.5+0.5
    
    # init camera and projector
    [dome_pi4node, dome_camera, dome_gpio]=init()
    
    # setup camera and start video preview
    set_camera_value('brightness', camera_bright_base, autoload=False)
    set_camera_value('framerate', 10, autoload=False)
    #set_camera_value('exposure comp', -18, autoload=False)
    load_camera_settings()
    open_camera()
    #dome_camera.show_info()
    
    # initialize projector
    screen_manager = DOMEproj.ScreenManager(proj_dim)
    update_pattern({"scale": scale}, prevent_log=True)
    update_pattern(off_light, prevent_log=True)
    
    # handle program termination
    atexit.register(terminate_session)
    signal.signal(signal.SIGTERM, terminate_session)
    signal.signal(signal.SIGINT,  terminate_session)
    signal.signal(signal.SIGABRT, terminate_session)
    signal.signal(signal.SIGILL,  terminate_session)
    signal.signal(signal.SIGSEGV, terminate_session)
    
    # start session
    print('Now adjust focus and camera parameters, test the calibration with validate_calibration(camera2projector) and then use start_experiment() to run the experiment.\n')
    
    