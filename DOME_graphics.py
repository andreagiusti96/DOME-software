# DOME_graphics.py
# #################################################################################################
# This code provides utility functions for plots and videos.
# #################################################################################################
# Authors = Andrea Giusti <andrea.giusti@unina.it>
# Affiliation = University of Naples Federico II
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os
from typing import List

import DOME_experiment_manager as DOMEexp

def highligth_inputs(inputs : np.array, time_instats = None, color='red', alpha : float = 0.3):
    inputs=np.ma.filled(inputs, fill_value=np.nan)
    input_differences = inputs[1:]-inputs[0:-1]
    input_differences[np.isnan(input_differences)]=0
    
    if len(time_instats)==0:
        time_instats = np.linspace(0, len(inputs)-1, len(inputs))

    on_value=max(max(input_differences), 1)
    off_value=min(min(input_differences), -1)
    
    
    ons  =np.where(input_differences ==on_value)[0]
    offs =np.where(input_differences ==off_value)[0]
    
    if len(ons) * len(offs) >0:
        if max(ons)>max(offs):
            offs = np.append(offs, np.where(~np.isnan(inputs))[0][-1])
        if min(offs)<min(ons):
            ons = np.append(ons, np.where(~np.isnan(inputs))[0][0])
            ons=sorted(ons)
    
    elif len(ons)==0 and len(offs) >0:
        ons = np.append(ons, np.where(~np.isnan(inputs))[0][0])
        ons=sorted(ons)
    
    elif len(offs)==0 and len(ons) >0:
        offs = np.append(offs, np.where(~np.isnan(inputs))[0][-1])
        offs=sorted(offs)
        
    else: 
        return
        
    for i in range(len(ons)):
        plt.axvspan(time_instats[ons[i]], time_instats[offs[i]], color=color, alpha=alpha, zorder=0)

def draw_trajectories(positions : np.array, contours : List = [], inactivity : np.array = np.zeros(0), img : np.array = np.zeros([1080, 1920]), title : str ="", max_inactivity : int = -1, time_window : int = -1, show:bool = True):
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
    
    # mark estimated or interpolated positions
    #est_positions = pos.copy().astype(float)
    #est_positions[inactivity==0] = np.nan
    plt.gca().set_prop_cycle(None)
    #plt.plot(pos[inactivity>0,0], pos[inactivity>0,1], 'x', markersize=10)
    
    for obj in range(pos.shape[1]):
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
        
# def process_img(img : np.array, color : str = "", blur  : int = 0, gain  : float = 1., contrast : bool =False, equalize : bool =False, plot : bool =False):
#    
#     if color == "gray":
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     elif color == "blue" or color == "b":
#         b, g, r = cv2.split(img)
#         img = b
#     elif color == "green" or color == "g":
#         b, g, r = cv2.split(img)
#         img = g
#     elif color == "red" or color == "r":
#         b, g, r = cv2.split(img)
#         img = r
#   
#     if plot: draw_image(img, "img - " + color)
#    
#     if gain != 1.0:
#         if gain < 0: 
#             min_val=img[640:1280, 360:720].min()
#             #img = cv2.convertScaleAbs(img, beta=-min_val)
#             img = cv2.max(float(min_val), img)
#             img = img-min_val
#             max_val=img[640:1280, 360:720].max()
#             gain=255.0/(max_val+0.1)
#            
#         img = cv2.convertScaleAbs(img, alpha= gain)
#         if plot: draw_image(img, "scaled")
#        
#     if equalize:
#         img = cv2.equalizeHist(img)
#         if plot: draw_image(img, "equalized")
#        
#     if contrast:
#         #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16,16))
#         img=clahe.apply(img)
#         if plot: draw_image(img, "contrasted")
#        
#     if blur:
#         img=cv2.medianBlur(img, blur)
#         if plot: draw_image(img, "blured")
#        
#     return img

# def build_background(fileLocation : str, images_number : int, gain : float = 1.0):
#     """
#     Extract the background from a set of images excluding moving objects.
#     Background is computed as the median pixel-wise of the images.
#     Camera has to be static.
    
#     Parameters
#     ----------
#     fileLocation : str
#         Path of the folder containing the images.
#     images_number : int
#         Number of images to use to build the background.

#     Returns
#     -------
#     background : np.array
#         Gray scale image of the background.

#     """
#     paths=glob.glob(fileLocation +  '/fig_*.jpeg')
#     paths = sorted(paths, key=lambda x:float(re.findall("(\d+.\d+)",x)[-1]))

#     images = np.ndarray([images_number, 1080, 1920], dtype=np.uint8)
#     indices = np.linspace(0, len(paths)-1, num=images_number, dtype=int)
#     #indices = np.linspace(0, images_number-1, num=images_number, dtype=int)

#     selected_paths = [paths[i] for i in indices]

#     counter=0
#     for filename in selected_paths:
#         img = cv2.imread(filename)
#         # Convert the frame to grayscale
#         elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)
#         #draw_image(elaborated_img, "img "+str(counter))
#         images[counter] = elaborated_img
#         counter+=1
    
#     # compute background
#     background = np.median(images, axis=0)
#     #background = np.mean(images, axis=0)
    

#     background = background.round().astype(np.uint8)
#     background = np.squeeze(background)
    
#     background = process_img(background, blur=DEFAULT_BLUR, gain=gain)

#     draw_image(background, "background from color "+DEFAULT_COLOR)
    
#     return background


# def get_contours(img : np.array, bright_thresh : List, area_r : List, compactness_r : List, background_model=None, expected_obj_number : int =0, plot : bool = False): 
#     """
#     Thresholding based object detection.

#     Parameters
#     ----------
#     img : np.array
#         Image to analyse.
#     min_area : float
#         Minimum area of objects in pixels.
#     min_compactness : float
#         Minimum compactness of objects [0, 1].
#     background_model : np.array, optional
#         Image of the background to perform background subtraction. 
#         The default is None.

#     Returns
#     -------
#     contoursFiltered : List
#         Contours of the detected objects.

#     """
#     contoursFiltered=[]
#     if plot: draw_image(img, "img")

#     elaborated_img = process_img(img, color=DEFAULT_COLOR, blur=DEFAULT_BLUR, gain=AUTO_SCALING)
#     if plot: draw_image(elaborated_img, "elaborated img from color " + DEFAULT_COLOR)

#     # Subtract the background from the frame
#     if type(background_model)==type(None): background_model= np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#     #foreground = cv2.min(cv2.subtract(elaborated_img, background_model), elaborated_img)
#     foreground = cv2.subtract(elaborated_img, background_model)
#     if plot: draw_image(foreground, "foreground")
    
#     foreground = process_img(foreground, blur=DEFAULT_BLUR)
#     if plot:  draw_image(foreground, "elaborated foreground")
    
#     threshold = bright_thresh[0]
#     a_r=area_r #.copy()
#     c_r=compactness_r #.copy()
    
#     first_time=True
#     margin_factor = 0.25
#     adjustment_factor = 0.02
    
#     too_few_obgs = False
#     too_many_obgs = False
    
#     # Perform brightness and shape thresholding.
#     # Iterate as long as the number of detected objects is not close to the expected one.
#     while first_time or (too_few_obgs and threshold > 55) or (too_many_obgs and threshold<200):
#         first_time=False
        
#         # Apply thresholding to the foreground image to create a binary mask
#         ret, mask = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)
#         if plot: draw_image(mask, "mask")
    
#         # Find contours of objects
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours_img=img.copy()
#         cv2.drawContours(contours_img, contours, -1, (0,255,0), 3)
        
#         # Select objects with the right size and compactness
#         contoursFiltered=[]
#         for i in range(len(contours)):
#             contour=contours[i]
#             area = cv2.contourArea(contour)
            
#             # print contours info
#             (Cx,Cy) = np.squeeze(get_positions(contours[i:i+1])).astype(int)
#             cv2.putText(contours_img, "A="+str(round(area)), (Cx+20,Cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,4)
            
#             if a_r[0] <= area <= a_r[1]:
#                 perimeter = cv2.arcLength(contour,True)
#                 compactness=(4*np.pi*area)/(perimeter**2) #Polsby–Popper test
#                 cv2.putText(contours_img, "C="+str(round(compactness,2)), (Cx+20,Cy+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) ,4)
                
#                 if c_r[0] <= compactness <= c_r[1]:
#                     contoursFiltered.append(contour)
        
        
#         if expected_obj_number==0: expected_obj_number = len(contoursFiltered)
        
#         if plot: draw_image(contours_img, "contours with thresh=" +str(threshold))
        
#         contoursFiltered_img=cv2.cvtColor(foreground,cv2.COLOR_GRAY2RGB)
#         cv2.drawContours(contoursFiltered_img, contoursFiltered, -1, (0,255,0), 3)
#         if plot: draw_image(contoursFiltered_img, "contoursFiltered with thresh=" +str(threshold))
        
#         # If the number of detected objects is not close to the expected one adjust the thresholds and iterate
#         print("thresh="+ str(round(threshold)) +"\t area_r="+ str(np.around(a_r)) +"\t compactness_r="+ str(np.around(c_r,2)) +"\t objects=" + str(len(contoursFiltered))+"\t exp objects=" + str(expected_obj_number))
#         too_many_obgs = len(contoursFiltered)-expected_obj_number >  np.ceil(expected_obj_number*margin_factor*2.0)
#         too_few_obgs = len(contoursFiltered)-expected_obj_number < - np.ceil(expected_obj_number*margin_factor)
        
#         if too_few_obgs :
#             bright_thresh[0]=bright_thresh[0] * (1-adjustment_factor)
#             threshold = bright_thresh[0]
#             a_r[0]=a_r[0] * (1-adjustment_factor)
#             a_r[1]=a_r[1] * (1+adjustment_factor)            
#             c_r[0]=c_r[0] * (1-adjustment_factor)
#             c_r[1]=c_r[1] * (1+adjustment_factor)
        
#         elif too_many_obgs :
#             bright_thresh[0]=bright_thresh[0] * (1+adjustment_factor)
#             threshold = bright_thresh[0]
#             a_r[0]=a_r[0] * (1+adjustment_factor)
#             a_r[1]=a_r[1] * (1-adjustment_factor)            
#             c_r[0]=c_r[0] * (1+adjustment_factor)
#             c_r[1]=c_r[1] * (1-adjustment_factor)
        
#     return contoursFiltered

# def get_positions(contours):
#     """
#     Get the centers of contours resulting from image analysis

#     Parameters
#     ----------
#     contours : List
#         Contours of the detected objects. (Shape=Nx2)

#     Returns
#     -------
#     positions : List
#         Position of the center of each object. (Shape=Nx2)

#     """
#     positions = []
#     for contour in contours:
#         (x,y,w,h) = cv2.boundingRect(contour)
#         positions.append([x+(w/2),y+(h/2)])
    
#     return positions

# def distance_from_edges(pos : List):
#     """
#     Get the distance from the closest edge of the picture frame.

#     Parameters
#     ----------
#     pos : List
#         Position. Shape=(2)

#     Returns
#     -------
#     dis : float or int depending on the input
#         Distances from the closest edge.

#     """
#     assert(len(pos)==2)
    
#     dis= np.min([pos[0], pos[1], 1920-pos[0], 1080-pos[1]])
    
#     assert(dis>=0)
#     return dis


# def valid_positions(positions : np.array):
#     """
#     Get the indices of valid positions, i.e. positions in the range [0, 1920][0, 1080]

#     Parameters
#     ----------
#     positions : np.array
#         Array of positions. Shape=(Nx2)

#     Returns
#     -------
#     validity : np.array
#         Array of bool telling whether the corresponding position is valid.

#     """
#     validity0=(positions[:,0] >= 0) & (positions[:,0] <= 1920)
#     validity1=(positions[:,1] >= 0) & (positions[:,1] <= 1080)
#     validity2= ~ np.isnan(positions[:,0])
#     validity = validity0 * validity1 * validity2
    
#     assert len(validity) == positions.shape[0]
#     return validity

def std_color_for_index(index : int):
    index = index%len(STD_COLORS)
    color = STD_COLORS[index]
    return color

# # CONSTANTS
# DEFAULT_COLOR = "red"
# DEFAULT_BLUR = 9
# AUTO_SCALING = -1

STD_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


