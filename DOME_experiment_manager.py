# DOME_experiment_manager.py
# #################################################################################################
# The ExperimentManager calss is provided to handle experiments and store the outputs, while
# keeping a timestamped log.
# #################################################################################################
# Authors = Andrea Giusti <andrea.giusti@unina.it>
# Affiliation = University of Naples Federico II
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import time
import os
import glob
import re
import cv2

from pathlib import Path
from datetime import datetime
import numpy as np


class ExperimentManager:
    '''
    Class for managing the data generated during DOME experiments.
    '''
    
    def __init__(self, date='', species='', culture ='', output_directory='/home/pi/Documents/experiments'):
        '''
        Sets up the details of the experiment.
        '''
        self.start_time = datetime.now()
        self.name=''
        self.path=''
        self.master_directory = output_directory
        
        if date == '' or date == 'today':
            date = datetime.today().strftime('%Y_%m_%d')
        
        if species != '' and date != '': 
            self.name = date + '_' + species
            self.path = os.path.join(output_directory, self.name)
            
            # create new folder
            counter=1
            while os.path.exists(self.path + f'_{counter}'):
                counter=counter+1
            
            self.name = self.name + f'_{counter}'
            self.path = os.path.join(output_directory, self.name)
            
            os.mkdir(self.path)
            
            with open(os.path.join(self.path, 'experiment_log.txt'), 'w') as file:
                file.write(f'Experiment \n\nDate={date} \nSpecies={species} \nCulture={culture} \nTrial={counter}\n\nCreated='+self.start_time.strftime('%H:%M:%S')+'\n')    
            
            with open(os.path.join(output_directory, 'experiments_list.txt'), 'a') as file:
                file.write(f'\n{date},'.ljust(11) + f'\t{species},'.ljust(12) + f'\t{culture},'.ljust(18) + f'\t{counter}')
    
    
    def __enter__(self):
        """
        Compatibility method to allow class to be used in "with" statements.
        ---
        Outputs
            self : ExperimentManager
                The instance of the ExperimentManager class.
        """
        return self
    
    
    def __exit__(self, type, value, traceback):
        '''
        Close ExperimentManager object upon exiting a "with" statement.
        ''' 
    
    def reset_starting_time(self, time=''):
        '''
        Reset starting time of the experiment.
        ---
        Parameters
            time=''
                String representing the starting time of the experiment
        '''
        
        if time=='':
            self.start_time=datetime.now()
        else:
            self.start_time=datetime.strptime(time, '%H:%M:%S')
    
    def save_data(self, title : str, force:bool=False, *args, **kwds):
        '''
        Save data in an .npz file in the experiment folder.
        This function can be called as:
            current_experiment.save_data(title="data",
                                 activation_times=activation_times,
                                 totalT = totalT)
        ---
        Parameters:
            title : str
                Name of the npz file.
            *args
                Data to be saved.
            **kwds
                Keywords to assign to the data.
        ---
        See also get_data.
        '''
        
        file_path = os.path.join(self.path, title + ".npz")
        
        if os.path.isfile(file_path):
            if force: 
                print(f'File {file_path} was overwritten\n')
            else:
                print(f'File {file_path} already exists!\n')
                return
        
        if self.name=='default':
            print('First create an experiment with new_experiment().\n')
            return
        
        np.savez(file_path, *args, **kwds)
        
    def get_data(self, title : str = 'data.npz', allow_pickle:bool = False):
        '''
        Read data from an .npz or .npy file in the experiment folder.
        ---
        Parameters:
            title : str = 'data.npz'
                Name of the npz or npy file.
        ---
        Outputs:
            data
                Data loaded from the file.
        ---
        See also save_data.
        '''
        
        file_path = os.path.join(self.path, title)
        
        if not os.path.exists(file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        data=np.load(file_path, allow_pickle=allow_pickle)
        return data

    def get_totalT(self):
        '''
        Get the total duration of the experiment in seconds.
        ---
        Outputs:
            totalT
                Total duration of the experiment in seconds.
        '''
        try:
            with self.get_data('data.npz') as data:
                totalT = data['totalT'] 
        except KeyError:
            dirr = os.path.join(self.path, 'images')
            paths = glob.glob(dirr +  '/*.jpeg')
            #paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
            totalT = max([get_time_from_title(f) for f in paths])
        return totalT
    
    def get_deltaT(self):
        '''
        Get the sampling time of the experiment in seconds.
        ---
        Outputs:
            deltaT
                Sampling time of the experiment in seconds.
        '''
        try:
            with self.get_data('data.npz') as data:
                deltaT = data['deltaT'] 
        except KeyError:
            dirr = os.path.join(self.path, 'images')
            paths = glob.glob(dirr +  '/*.jpeg')
            paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
            deltaT = get_time_from_title(paths[1]) - get_time_from_title(paths[0])
        return deltaT
    
    def get_img_at_time(self, image_time : float):
        '''
        Get the image captured from the camera at a given time instant.
        ---
        Parameters:
            time:float
                Time instant to get the image.
        ---
        Outputs:
            img : np.ndarray
                Image.
        '''
        paths=glob.glob(os.path.join(self.path, 'images') +  '/*.jpeg')
        paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
        
        for filename in paths:
            if get_time_from_title(filename) == image_time:
                img = cv2.imread(filename)
                break
        
        assert os.path.isfile(filename), f'Image {filename} not found!'
        assert isinstance(img, np.ndarray), f'{filename} is not a valid image!\nType is {type(img)}'
        return img

    def get_pattern_at_time(self, time:float):
        '''
        Get the projected pattern at a given time instant.
        ---
        Parameters:
            time:float
                Time instant to get the pattern.
        ---
        Outputs:
            pattern : np.ndarray
                Projected pattern image.
        '''
        pattern = None
        
        if os.path.isdir(os.path.join(self.path, 'patterns_cam')):
            files = glob.glob(os.path.join(self.path, 'patterns_cam') +  '/*.jpeg')
            files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
            file = [f for f in files if get_time_from_title(f) <= time][-1]
            pattern=cv2.imread(file)
        elif os.path.isdir(os.path.join(self.path, 'patterns')):
            files = glob.glob(os.path.join(self.path, 'patterns') +  '/*.jpeg')
            files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
            file = [f for f in files if get_time_from_title(f) <= time][-1]
            pattern=cv2.imread(file)
        else:
            with self.get_data('data.npz') as data:
                 patterns = data['patterns']
                 pattern = patterns[int(time*2)]
                 
        return pattern
        
    def add_detail(self, message : str, include_in_exp_list : bool =False):
        '''
        Add detail to the experiment log file.
        ---
        Parameters
            message : str
                Message to add in the log file.
            include_in_exp_list : bool =False
                If True the message is also added in the experiments_list.txt file.
        '''
        
        if self.name=='default':
            print('First create an experiment with new_experiment().\n')
            return
        
        file_path = os.path.join(self.path, 'experiment_log.txt')
        
        if not os.path.exists(file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        with open(file_path, 'a') as file:
            file.write('\n' + message)
        
        if not include_in_exp_list:
            return
        
        list_file_path = os.path.join(self.master_directory, 'experiments_list.txt')
        
        if not os.path.exists(list_file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        with open(list_file_path, 'a') as file:
            file.write(',\t' + message)   
    
    def add_log_entry(self, message : str):
        '''
        Add time stamped entry to the experiment log.
        ---
        Parameters
            message : str
                Message to append
        '''
        ellapsed_time= datetime.now() - self.start_time
        file_path = os.path.join(self.path, 'experiment_log.txt')
        
        if self.name=='default':
            print('First create an experiment with new_experiment().\n')
            return
        
        if not os.path.exists(file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        with open(file_path, 'a') as file:
            timestamp=str(ellapsed_time)
            microseconds=timestamp.split('.')[1]
            timestamp=timestamp.split('.')[0]
            timestamp=timestamp+'.'+microseconds[0]
            if ', ' in timestamp:
                timestamp=timestamp.split(', ')[1]
            file.write('\n' + timestamp + ', ' + message)    
    
    def get_trackings(self):
        paths = glob.glob1(self.path, 'tracking_*')
        if len(paths)==0:
            print(f'No tracking found for experiment {self.name}! Use DOME_tracker to generate one.')
        else:
            print(paths)
        return paths
    
    def get_last_tracking(self):
        paths = glob.glob1(self.path, 'tracking_2*')
        assert len(paths)>0, f'No tracking found for experiment {self.name}! Use DOME_tracker to generate one.'
        last_tracking = paths[-1]
        return last_tracking
        


# other functions
def open_experiment(experiment_name : str, output_directory:str ='/home/pi/Documents/experiments'):
    '''
    Start working in existing experiment folder.
    ---
    Parameters
        experiment_name : str
            Name of the experiment to open.
        output_directory:str ='/home/pi/Documents/experiments'
            Containing directory.
    ---
    Outputs:
        experiment:ExperimentManager
            Opened experiment.
    '''
    path = os.path.join(output_directory, experiment_name)
    
    if os.path.exists(path):
        experiment = ExperimentManager()
        experiment.name=experiment_name
        experiment.path=path
        print(f'Now working in {path}.')
    else:
        raise(Exception(f'{path} not found!\n'))
    
    return experiment

def get_time_from_title(filename: str):
    """
    Extract time from a string.
    ---
    Parameters:
        filename : str
            String to extract the time from.
    ---
    Outputs:
        time : float
            Time extracted from the string.
    """
    filename = filename.split("fig_")[-1]
    filename = filename.split("pattern_")[-1]
    filename = filename.split(".jpeg")[0]
    time = float(filename)
    return time
    