#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOME_graphics.py
This code provides utility functions for plots and videos.

See also: DOME_experiment_manager and DOME_imaging_utilities.

Author:     Andrea Giusti
Created:    2023
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import os
from typing import List

import DOME_experiment_manager as DOMEexp

def highligth_inputs(inputs : np.array, time_instants = None, color='red', alpha : float = 0.3, axis = None):
    inputs=np.ma.filled(inputs, fill_value=np.nan)
    input_differences = inputs[1:]-inputs[0:-1]
    input_differences[np.isnan(input_differences)]=0
    
    if len(time_instants)==0:
        time_instants = np.linspace(0, len(inputs)-1, len(inputs))
        
    if not axis:
        axis=plt.gca()

    on_value=max(max(input_differences), 1)
    off_value=min(min(input_differences), -1)
    
    
    ons  =np.where(input_differences ==on_value)[0]
    offs =np.where(input_differences ==off_value)[0]
    
    if len(ons) * len(offs) >0:
        if max(ons)>max(offs):
            offs = np.append(offs, np.where(~np.isnan(inputs))[0][-1])
        if min(offs)<min(ons):
            ons = np.append(ons, np.where(~np.isnan(inputs))[0][0])
            
    
    elif len(ons)==0 and len(offs) >0:
        ons = np.append(ons, np.where(~np.isnan(inputs))[0][0])
    
    elif len(offs)==0 and len(ons) >0:
        offs = np.append(offs, np.where(~np.isnan(inputs))[0][-1])
        
    else: 
        return
    
    for i in range(len(ons) - len(offs)):
        offs = np.append(offs, np.where(~np.isnan(inputs))[0][-1])
        
    ons=sorted(ons)
    offs=sorted(offs)
    
    for i in range(len(ons)):
        axis.axvspan(time_instants[ons[i]], time_instants[offs[i]], color=color, alpha=alpha, zorder=0)

def draw_trajectories(positions : np.array, contours : List = [], inactivity : np.array = np.zeros(0), 
                      img : np.array = np.zeros([1080, 1920]), title : str ="", max_inactivity : int = -1, 
                      time_window : int = -1, show:bool = True):
    
    fig = plt.figure(1,figsize=(19.20,10.80),dpi=100)
    fig.subplots_adjust(top=1.0-0.05, bottom=0.05, right=1.0-0.05, left=0.05, hspace=0, wspace=0) 
    plt.title(title); 
    
    if inactivity.shape[0] == 0:
        inactivity = np.zeros(positions.shape[0:2])
    
    if isinstance(inactivity, List):
        inactivity=np.array(inactivity)

    if len(img.shape)==2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    # get index of last valid position (discard future positions)
    counter = inactivity.shape[0] - (inactivity[:,0] >= 0)[::-1].argmax(0) -1
    pos = positions.copy()
    inac = inactivity.copy()
    
    # discard longly inactive objects
    if max_inactivity >= 0:
        obsolete_obgs = inac[counter,:] > max_inactivity
        pos[:,obsolete_obgs,:]=np.nan

    # select recent data
    if time_window > 0:
        time_window_start = max([counter+1-time_window, 0])
        #pos[:time_window_start,:,:]=np.nan
        pos=pos[time_window_start:,:,:]
        inac = inac[time_window_start:, :]      
    
    # Plot trajectories
    plt.plot(pos[:,:,0],pos[:,:,1],'o-', markersize=3)
    

    plt.gca().set_prop_cycle(None)
    
    for obj in range(pos.shape[1]):
        # mark estimated or interpolated positions
        plt.plot(pos[inac[:,obj]>0,obj,0], pos[inac[:,obj]>0,obj,1], 'x', markersize=10)
        last_index = pos.shape[0] - (~np.isnan(pos[:,obj,0]))[::-1].argmax(0) -1
        if not np.isnan(pos[last_index,obj,0]):
            plt.text(pos[last_index,obj,0], pos[last_index,obj,1], str(obj), fontsize = 22, color = std_color_for_index(obj))
            if len(contours)>0:
                contour =np.array(contours[:][obj][:]).squeeze()
                plt.plot(contour[:,0],contour[:,1], color = std_color_for_index(obj))
    
    plt.xlim([0, 1920])
    plt.ylim([1080, 0])
    plt.xticks(range(0,1921,480))
    plt.yticks(range(0,1081,270))

    if show:
        plt.show()
    else:
        plt.close()

    return fig

def draw_image(img : np.array = np.zeros([1080, 1920]), title : str =""):
    plt.figure(1,figsize=(20,20),dpi=72)
    plt.title(title); 
    
    if len(img.shape)==2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    plt.xlim([0, 1920])
    plt.ylim([1080, 0])
    plt.xticks(range(0,1921,480))
    plt.yticks(range(0,1081,270))
    
    plt.show(block=False)

def histogram(img : np.array):
    plt.title("Histogram");
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    
def make_video(directory : str, title : str = "video.mp4", fps : float = 1, key : str = '/*.jpeg'):
    assert os.path.isdir(directory), f'Directory {directory} not found.'

    paths = glob.glob(directory + key)
    paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))

    assert len(paths)>0, 'No images found. Video cannot be generated!'
    
    dim=(1920,1080)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
    
    video = cv2.VideoWriter()
    
    succes = video.open(os.path.join(directory,title), fourcc, fps, dim, True)
    assert succes, 'Video creation failed! Try replacing your installation of opencv with opencv-python.'
    
    i=0;
    for path in paths:
        frame=cv2.imread(path)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        video.write(resized)
        i+=1; print(f'\rGenerating video from {len(paths)} images: {round(i/len(paths)*100,1)}%', end='\r')
        
    cv2.destroyAllWindows()
    video.release()
    print(f'\nVideo {title} saved in {directory}')

def overlap_pattern(experiment, time:float, alpha:float=1):
    img = experiment.get_img_at_time(time)
    #draw_image(img,'img')
    pat = experiment.get_pattern_at_time(time)
    pat = cv2.resize(pat, [img.shape[1], img.shape[0]])
    pat = cv2.convertScaleAbs(pat, alpha=alpha)
    #draw_image(pat,'pat')
    img = cv2.add(img, pat)
    return img
    
def overlap_patterns(experiment, alpha:float=1, save_fig:bool=False, save_video:bool=True, fps:float=1):    
    dim=(1920,1080)
    
    files = glob.glob(os.path.join(experiment.path, 'images') +  '/*.jpeg')
    files = sorted(files, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))
    
    totalT = experiment.get_totalT()
    
    if save_fig:
        output_folder = 'images_pat'
        output_dir = os.path.join(experiment.path, output_folder)
        try:
            os.mkdir(output_dir)
        except OSError:
            pass
    
    if save_video:
        video_name = f'{experiment.name}_pat.mp4'
        video_path = os.path.join(experiment.path,video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, frameSize=dim)
    
    for f in files:
        t = DOMEexp.get_time_from_title(f)
        print(f'\r{round(t/totalT*100,1)}%', end='\r')
        new_img = overlap_pattern(experiment, time=t, alpha=alpha)
        if save_fig: cv2.imwrite(os.path.join(output_dir,'fig_' + '%04.1f' % t + '.jpeg'), new_img)
        if save_video: video.write(new_img)
    print('')
   
    cv2.destroyAllWindows()
    if save_video:
        video.release()
        print(f'Video saved as {video_name}')
    if save_fig:
        print(f'Images saved in {output_dir}')
        

def cropCmap(cmap_name : str, minVal:float, maxVal:float):
    try:
        cmap = plt.colormaps[cmap_name]
    except TypeError:
        cmap = plt.get_cmap(cmap_name)
    
    new_cmap = LinearSegmentedColormap.from_list('New'+cmap_name, cmap(np.linspace(minVal, maxVal, 10)), N=256, gamma=1.0)
    return new_cmap

def std_color_for_index(index : int):
    index = index%len(STD_COLORS)
    color = STD_COLORS[index]
    return color

STD_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


