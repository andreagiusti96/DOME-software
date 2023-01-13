# The ExperimentManager calss is provided to handle experiments and store the outputs, while
# keeping a timestamped log.
# The DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio Denniss. This code requires
# the "DOME_caibration_projector.py" file to be run in parallel on the Raspberry Pi 0 connected to
# the DOME projector.
# #################################################################################################
# Authors = Andrea Giusti <andrea.giusti@unina.it>
# Affiliation = University of Naples Federico II
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import time
import os
from pathlib import Path
from datetime import datetime
from datetime import date
import numpy as np


class ExperimentManager:
    '''
    Class for managing DOME experiments.
    '''
    
    def __init__(self, date='', species='', culture ='', output_directory='/home/pi/Documents/experiments'):
        '''
        Sets up the details of the experiment.
        '''
        self.start_time = datetime.now()
        self.name=''
        self.path=''
        
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
                file.write(f'{date},\t{species},\t{culture},\t{counter}\n')    
    
    
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
    
    def save_data(self, *args, **kwds):
        '''
        Add detail to the experiment log.
        ---
        Parameters
        '''
        file_path = os.path.join(self.path, 'data.npz')
        
        if self.name=='default':
            print('First create an experiment with new_experiment().\n')
            return
        
        np.savez(file_path, *args, **kwds)
        
    def get_data(self):
        '''
        Add detail to the experiment log.
        ---
        Parameters
        '''
        file_path = os.path.join(self.path, 'data.npz')
        
        if not os.path.exists(file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        data=np.load(file_path)
        return data

    
    def add_detail(self, message : str):
        '''
        Add detail to the experiment log.
        ---
        Parameters
            message : str
                message to append
        '''
        file_path = os.path.join(self.path, 'experiment_log.txt')
        
        if self.name=='default':
            print('First create an experiment with new_experiment().\n')
            return
        
        if not os.path.exists(file_path):
            raise(Exception(f'{file_path} not found !\n'))
        
        with open(file_path, 'a') as file:
            file.write('\n' + message)
    
    def add_log_entry(self, message : str):
        '''
        Add time stamped entry to the experiment log.
        ---
        Parameters
            message : str
                message to append
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

# other functions
def open_experiment(experiment_name : str, output_directory='/home/pi/Documents/experiments'):
    '''
    Start working in existing experiment folder.
    ---
    Parameters
        experiment_name : str
            Name of the experiment to open
    '''
    path = os.path.join(output_directory, experiment_name)
    
    if os.path.exists(path):
        experiment = ExperimentManager()
        experiment.name=experiment_name
        experiment.path=path
        print(f'Now working in {path} \nRead the starting time from the log file and set it using reset_starting_time().\n')
    else:
        print(f'{path} not found !\n')
    
    return experiment
