# This code is provided to analyse the results of a DOME experiment. 
# See DOME_experiment.py to launch an experiment.
# The DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio Denniss.
# #################################################################################################
# Authors = Andrea Giusti <andrea.giusti@unina.it>
# Affiliation = University of Naples Federico II
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

#import DOME_communication as DOMEcomm
#import DOME_imaging_utilities as DOMEutil
import DOME_experiment_manager as DOMEexp

import numpy as np
#import matplotlib
import cv2
import json
import time
import os
from pathlib import Path
from datetime import datetime
from datetime import date

def load_experiment(name : str, output_directory='/home/pi/Documents/experiments'):
    '''
    Access an existing experiment and load the data from the data.npz archive.
    Parameters
    ----------
    name : str
        Name of the experiment.
    output_directory : str, optional
        Directory where the experiment folder is located. 
        The default is '/home/pi/Documents/experiments'.

    Returns
    -------
    current_experiment : ExperimentManager
        Opened experiment.
    '''
    
    current_experiment= DOMEexp.open_experiment(name, output_directory)    
    print(f'Now working in folder {current_experiment.path}\n')
    
    return current_experiment


if __name__ == '__main__':
    
    # details of the experiment    
    #output_directory='/home/pi/Documents/experiments'
    output_directory='/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
    experiment_name='2022_12_16_Prova_17'
    
    # retrive experiment
    current_experiment=load_experiment(experiment_name, output_directory)
    
    # read data
    data = current_experiment.get_data()
    activation_times = np.copy(data['activation_times'])
    images = np.copy(data['images'])
    patterns = np.copy(data['patterns'])
    
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow('preview',images[1])
    cv2.waitKey()

    
