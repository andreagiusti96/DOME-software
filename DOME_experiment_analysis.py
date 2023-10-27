#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created by Andrea

@author: andrea
"""


import cv2
import numpy as np
import pandas as pd
import scipy
import glob
import matplotlib.pyplot as plt
#import plotly.express as px
import seaborn as sns
import sys
import os
import re
import random
from typing import List
import DOME_experiment_manager as DOMEexp
import DOME_graphics as DOMEgraphics
import DOME_tracker as DOMEtracker

def MLE_SDE_parameters(x : np.array, deltaT : float):
    '''
    Estimate parameters of a SDE in the form
        dX = theta * (mu - X) * dt + sigma * dW
    where dW is gaussian white noise.
    
    The algorithm is described in "Calibrating the Ornstein-Uhlenbeck (Vasicek) model" by van den Berg (2011).

    Parameters
    ----------
    x : np.array
        vector of equally spaced data.
    deltaT : float
        sampling time step.

    Returns
    -------
    mu : float
        Estimated mean.
    theta : float
        Estimated reversion rate.
    sigma : float
        Estimated standard deviation of the noise.
    '''
    assert len(x.shape) == 1, "x must be one-dimensional."
    
    #convert masked array and esclude nans
    x=np.ma.filled(x, fill_value=np.nan)
    x = x[~np.isnan(x)]
    
    n=len(x)-1
    
    # compute moments
    s_x = np.sum(x[:-1])
    s_y = np.sum(x[1:])
    s_xx = np.sum(x[:-1]**2)
    s_yy = np.sum(x[1:]**2)
    s_xy = np.sum(x[:-1] * x[1:])
    
    # compute estimated parameters
    mu = (s_y*s_xx - s_x*s_xy) / (n *(s_xx - s_xy) - (s_x**2 - s_x*s_y))
    theta = -1/deltaT * np.log((s_xy - mu*s_x - mu*s_y + n*mu**2)/(s_xx - 2*mu*s_x + n*mu**2))
    
    a = np.exp(-theta*deltaT)
    sigma = np.sqrt(2*theta/(1-a**2) * 1/n * (s_yy - 2*a*s_xy + a**2*s_xx - 2*mu*(1-a)*(s_y - a*s_x) + n*mu**2*(1-a)**2))
    
    return mu, theta, sigma
    
def identify_agents_parameters(experiment : DOMEexp.ExperimentManager, agents:list=[]):
    deltaT = experiment.get_deltaT()
    analysed_data_path = os.path.join(experiment.path, experiment.get_last_tracking(), "analysed_data.npz")
    with experiment.get_data(analysed_data_path) as analysed_data:
        speeds_smooth=analysed_data["speeds_smooth"]
        speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))
        ang_vel_smooth=analysed_data["ang_vel_smooth"]
        ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
    
    if len(agents)==0:
        agents = list(range(speeds_smooth.shape[1]))
    
    number_of_agents = len(agents)
    
    mu_s    = [np.nan] * number_of_agents
    theta_s = [np.nan] * number_of_agents
    sigma_s = [np.nan] * number_of_agents
    mu_w    = [np.nan] * number_of_agents
    theta_w = [np.nan] * number_of_agents
    sigma_w = [np.nan] * number_of_agents
    
    for agent in range(number_of_agents):
        mu_s[agent], theta_s[agent], sigma_s[agent] = MLE_SDE_parameters(speeds_smooth[:,agents[agent]], deltaT)
        mu_w[agent], theta_w[agent], sigma_w[agent] = MLE_SDE_parameters(ang_vel_smooth[:,agents[agent]], deltaT)
        
    experiment.save_data_txt(title='identification', data=[agents, mu_s, theta_s, sigma_s, mu_w, theta_w, sigma_w], 
                             force=True, labels=['ids', 'mu_s', 'theta_s', 'sigma_s', 'mu_w', 'theta_w', 'sigma_w'])
        
    #return mu_s, theta_s, sigma_s, mu_w, theta_w, sigma_w
    
def split(data : np.array, condition : np.array):
    out_data=[]
    out_data.append(data[condition])
    out_data.append(data[~condition])
    
    return out_data

def detect_outliers(data, m = 2., side='both'):
    assert side in ['both', 'top', 'bottom']
    
    data=np.ma.array(data, mask=np.isnan(data))
    d = np.abs(data - np.ma.median(data))
    mdev = np.ma.median(d)
    #s = d/mdev if mdev else np.zeros(len(d))
    s = d/mdev
    outliers=s>m
    
    if side=='top':
        outliers = outliers * (data > np.ma.median(data))
    elif side=='bottom':
        outliers = outliers * (data < np.ma.median(data))    
    
    return outliers

def remove_agents(agents : [int, List]):
    global speeds_smooth, acc_smooth, velocities, interp_positions
    
    if type(agents) is int:
        agents = [agents]
        
    for a in agents:
        speeds_smooth[:,a]=np.nan
        speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))
        acc_smooth[:,a]=np.nan
        acc_smooth = np.ma.array(acc_smooth, mask=np.isnan(acc_smooth))
        velocities[:,a]=np.nan
        interp_positions[:,a,:]= np.nan
        
    # # Save analysis data
    # current_experiment.save_data(os.path.join(tracking_folder, 'analysed_data'), force=True, time_steps=time_steps, 
    #                               interp_positions=interp_positions, speeds_smooth=speeds_smooth,
    #                               ang_vel_smooth=ang_vel_smooth, speeds_on=speeds_on, speeds_off=speeds_off,
    #                               ang_vel_on=ang_vel_on, ang_vel_off=ang_vel_off)

def detect_tumbling(speed, ang_vel, m=2.):
    speed=np.ma.array(speed, mask=np.isnan(speed))
    ang_vel=np.ma.array(ang_vel, mask=np.isnan(ang_vel))
    slow = detect_outliers(speed[:-1], m, 'bottom')
    turning = detect_outliers(ang_vel, m, 'top')
    tumbling = slow * turning
    return tumbling

def autocorrelation(x, axis=0):
    x=np.ma.filled(x, fill_value=np.nan)
    acorrs=[]
    
    if len(x.shape)>1:
        series = x.shape[axis-1]
    else:
        series=1
    
    for i in range(series):
        if len(x.shape)>1:
            data = np.take(x, i, axis-1)
        else:
            data=x
        
        if any(~np.isnan(data)):
            valid_data=data[~np.isnan(data)]
            mean = np.mean(valid_data)
            var = np.var(valid_data)
            ndata = valid_data - mean
            values=np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 

            acorr = values[~np.isnan(values)]
            
            acorr = acorr / var / len(ndata)

            acorrs.append(acorr)
        else:
            acorrs.append(np.array([]))
    
    return acorrs

def correlation(x, y, axis=0):
    x=np.ma.filled(x, fill_value=np.nan)
    y=np.ma.filled(y, fill_value=np.nan)
    corrs=[]
    
    if len(x.shape)>1:
        series = x.shape[axis-1]
    else:
        series=1
    
    for i in range(series):
        if len(x.shape)>1:
            data = np.take(x, i, axis-1)
        else:
            data=x
        
        if any(~np.isnan(data)):
            valid_data=data[~np.isnan(data)]
            mean = np.mean(valid_data)
            var = np.var(valid_data)
            ndata = valid_data - mean
            values=np.correlate(ndata, y, 'full')[len(ndata)-1:] 

            corr = values[~np.isnan(values)]
            
            corr = corr / var / len(ndata)

            corrs.append(corr)
        else:
            corrs.append(np.array([]))
    
    return corrs

def vector_autocorrelation(data):
    data=np.ma.filled(data, fill_value=np.nan)
    valid_data=data[~np.isnan(data[:,0])]   
    
    if len(valid_data)==0:
        return np.array([])
        
    n_row = valid_data.shape[0]
    dot_mat = valid_data.dot(valid_data.T).astype(float)
    norm_mat = np.outer(np.linalg.norm(valid_data, axis=1), np.linalg.norm(valid_data, axis=1))
    dot_mat /= norm_mat
    corr = [np.trace(dot_mat,offset=x) for x in range(n_row)]
    corr/=(n_row-np.arange(n_row))
    corr/=corr[0]
    return corr

def vector_correlation(data1, data2):
    data1=np.ma.filled(data1, fill_value=np.nan)
    data2=np.ma.filled(data2, fill_value=np.nan)
    valid_data1=data1[~np.isnan(data1[:,0])]   
    valid_data2=data2[~np.isnan(data2[:,0])]   
    
    if len(valid_data1)==0 or len(valid_data2)==0:
        return np.array([])
        
    corr_mat = valid_data1.dot(valid_data2.T).astype(float)
    norm_mat = np.outer(np.linalg.norm(valid_data1, axis=1), np.linalg.norm(valid_data2, axis=1))
    corr_mat /= norm_mat
    return corr_mat

def vector_covariance(data1, data2):
    data1=np.ma.filled(data1, fill_value=np.nan)
    data2=np.ma.filled(data2, fill_value=np.nan)
    valid_data1=data1[~np.isnan(data1[:,0])]   
    valid_data2=data2[~np.isnan(data2[:,0])]   
    
    if len(valid_data1)==0 or len(valid_data2)==0:
        return np.array([])
        
    cov_mat = valid_data1.dot(valid_data2.T).astype(float)
    return cov_mat

def vector_similarity(data1, data2):
    data1=np.ma.filled(data1, fill_value=np.nan)
    data2=np.ma.filled(data2, fill_value=np.nan)
    valid_data1=data1[~np.isnan(data1[:,0])]   
    valid_data2=data2[~np.isnan(data2[:,0])]   
    
    if len(valid_data1)==0 or len(valid_data2)==0:
        return np.array([])
        
    cov_mat = valid_data1.dot(valid_data2.T).astype(float)
    norms1 = np.linalg.norm(valid_data1, axis=1)
    norms2 = np.linalg.norm(valid_data2, axis=1)
    # norms_mat1 = np.outer(norms1, norms1)
    # norms_mat2 = np.outer(norms2, norms2)
    # norms_mat = np.max([norms_mat1, norms_mat2], axis=0)
    norms_mat = np.zeros([len(valid_data1), len(valid_data2)])
    for i in range(len(valid_data1)):
        for j in range(len(valid_data2)):
            norms_mat[i,j]=max(norms1[i], norms2[j])**2
    sim_mat = cov_mat / norms_mat
    return sim_mat

def vector_auto_similarity(data):
    data=np.ma.filled(data, fill_value=np.nan)
    valid_data=data[~np.isnan(data[:,0])]   
    
    if len(valid_data)==0:
        return np.array([])
        
    cov_mat = valid_data.dot(valid_data.T).astype(float)
    norms = np.linalg.norm(valid_data, axis=1)
    # norms_mat1 = np.outer(norms1, norms1)
    # norms_mat2 = np.outer(norms2, norms2)
    # norms_mat = np.max([norms_mat1, norms_mat2], axis=0)
    norms_mat = np.zeros([len(valid_data), len(valid_data)])
    for i in range(len(valid_data)):
        for j in range(len(valid_data)):
            norms_mat[i,j]=max(norms[i], norms[j])**2
    sim_mat = cov_mat / norms_mat
    sim_vec = [np.trace(sim_mat,offset=x) for x in range(len(valid_data))]
    sim_vec/=(len(valid_data)-np.arange(len(valid_data)))
    return sim_vec

def lag_auto_similarity(data, lag=1):
    auto_similarity = vector_similarity(data, data)
    if len(auto_similarity)>0:
        diag=np.diagonal(auto_similarity, offset=lag)
        lag_sim1 = np.ma.concatenate([diag, [diag[-1]]])
        lag_sim2 = np.ma.concatenate([[diag[0]], diag])
        lag_sim=np.mean([lag_sim1, lag_sim2], axis=0)
    else:
        lag_sim = np.array([])
    
    lag_similarity = data[:,0].copy()
    lag_similarity[~np.isnan(lag_similarity)] = lag_sim
    return lag_similarity
    
def moving_average(x, window, weights=[], axis=0):
    #x=np.array(x).astype(float)
    x=np.ma.filled(x, fill_value=np.nan)
    
    if weights == []:
        weights=np.ones(window)
    else:
        weights=np.array(weights)
    
    kernel=weights/sum(weights)
    y=np.zeros_like(x) * np.nan
    
    for i in range(x.shape[axis-1]):
        #y[:,i]= np.convolve(x[:,i], kernel, 'same')
        data = np.take(x, i, axis-1)
        out_data = data.copy();
        
        if sum(~np.isnan(data)) >= window:
            values=np.convolve(data, kernel, 'valid');
            edge=round(np.floor(window/2))

            (out_data[edge:-edge])[~np.isnan(values)] = values[~np.isnan(values)]
            
        if axis==0:
            y[:,i]=out_data
        elif axis==1:
            y[i,:]=out_data
    
    y=np.ma.array(y, mask=np.isnan(y))
    return y

def angle_diff(unit1, unit2):
    angles=np.zeros_like(unit1)
    
    for i in range(unit1.shape[0]):
        for j in range(unit1.shape[1]):
            angle1=unit1[i,j]
            angle2=unit2[i,j]
            
            if not np.isnan(angle1) and not np.isnan(angle2):
                phi = abs(angle2-angle1) % (2 * np.pi)
                sign = 1
                # calculate sign
                if not ((angle1-angle2 >= 0 and angle1-angle2 <= np.pi) or (
                        angle1-angle2 <= - np.pi and angle1-angle2 >= -2 * np.pi)):
                    sign = -1
                if phi > np.pi:
                    result = 2 * np.pi-phi
                else:
                    result = phi
            
                angles[i,j]=result*sign
            else:
                angles[i,j]=np.nan
            
    return angles
    
def my_histogram(data : np.array, bins=10, normalize=False):
    #data=np.array(data)
    
    number_of_series=len(data)
    
    if type(bins)==int:
        n_bins=bins
    else:
        n_bins=len(bins)-1
    
    values=np.zeros([number_of_series, n_bins])
    
    for i in range(number_of_series):
        val, bins = np.histogram(data[i], bins)
    
        if normalize:
            val = val/sum(val)
        
        values[i]=val
        
        positions=bins[:-1] + (bins[1:]-bins[:-1])/(number_of_series+1)*(i+1)
        plt.bar(positions, val, width=0.8*np.diff(bins)/number_of_series)
    
    plt.xticks(bins, np.round(bins,1))
    plt.xlim([min(bins), max(bins)])

def scatter_hist(x : np.ndarray, y : np.ndarray, c : np.ndarray = None, n_bins : int = 10, cmap = "viridis"):
    
    fig = plt.gcf()
    
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()

    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    number_of_series=len(x)
    
    for i in range(number_of_series):
        x_vec = np.ma.filled(x[i], fill_value=np.nan)
        y_vec = np.ma.filled(y[i], fill_value=np.nan)
        
        # the scatter plot
        if c is None:
            ax.scatter(x_vec, y_vec)
        else:
            c_vec = np.ma.filled(c[i], fill_value=np.nan)
            ax.scatter(x_vec, y_vec, c=c_vec, cmap = cmap)
        
        # the histograms     
        x_val, x_bins = np.histogram(x_vec[~np.isnan(x_vec)], n_bins)
        y_val, y_bins = np.histogram(y_vec[~np.isnan(y_vec)], n_bins)
        
        x_val = x_val/sum(x_val)/(x_bins[1]-x_bins[0])
        y_val = y_val/sum(y_val)/(y_bins[1]-y_bins[0])
       
        ax_histx.bar(x_bins[:-1], x_val, width=1.0*np.diff(x_bins), align='edge', alpha=0.5)
        ax_histy.barh(y_bins[:-1], y_val, height=1.0*np.diff(y_bins), align='edge', alpha=0.5)

    return ax

def my_boxplot(data : List, compare : str = 'no', whis=1.5):
    assert compare in ['no', 'within', 'across']
    
    if not type(data[0]) is list:
        data=[data]
    
    number_of_series=len(data)
    y=np.ma.max([np.ma.max(item) for row in data for item in row])
    separation = (number_of_series-1.5) * 1.5
    for i in range(number_of_series):
        data_to_plot = list(map(lambda X: [x for x in X if x], data[i]))
        
        positions=1+i+np.linspace(0, len(data_to_plot)+separation, len(data_to_plot))
        plt.boxplot(data_to_plot, positions=positions, patch_artist=True, 
                    boxprops=dict(facecolor=DOMEgraphics.std_color_for_index(i)),
                    medianprops=dict(color="black",linewidth=1), whis=whis)
        if compare == 'within' and len(data_to_plot)>1: 
            add_significance_bar(data=data_to_plot,positions=positions,y_base=y, delta_y=i*5+1, 
                                 rel_h=0.03, median_distance=True)
    
    if compare == 'across':
        for i in range(len(data[0])):
            for j in range(number_of_series):
                for k in range(j+1,number_of_series):
                    data_to_plot = list(map(lambda X: [x for x in X if x], [data[j][i], data[k][i]]))
                    positions = [i*(separation+2)+1+j, i*(separation+2)+1+k]
                    add_significance_bar(data=data_to_plot,positions=positions, adjust_y=False, 
                                         y_base=y, delta_y=(k+(k-j)-2)*5+1, rel_h=0.03, median_distance=True)
    
    if len(data[0]) > 1: 
        plt.xticks(np.linspace(0, len(data_to_plot)+separation, len(data_to_plot))+number_of_series/2+0.5)
    plt.ylim(top=y*(1.1+0.125*number_of_series))


def add_significance_bar(p_value=None, y_base=None, delta_y=1, data=None, median_distance=False,
                         positions=[1, 2], rel_h=0.05, use_stars=False, axis=None, adjust_y=True):
        
    if not p_value:
        #res = scipy.stats.ttest_ind(data[0], data[1], equal_var = False)
        #res = scipy.stats.ranksums(data[0], data[1])
        res = scipy.stats.mannwhitneyu(data[0], data[1])
        #res = scipy.stats.kstest(data[0], data[1])
        p_value = res.pvalue

    if not y_base:
        y_base = max(max(data[0]), max(data[1]))

    y = y_base * (1+rel_h*delta_y)
    h = y * rel_h
    col = 'k'

    if use_stars:
        text = '*' * int(-np.ceil(np.log10(p_value)))
    else:
        text = 'p=' + str(np.round(p_value, 3))
        if p_value > 0.05:
            text = 'ns'
        elif p_value < 0.001:
            text = 'p<0.001'
            
    if median_distance:
        med0 = np.ma.median(data[0])
        med1 = np.ma.median(data[1])
        med_dist = np.ma.abs(med0 - med1)
        text = (f'd=%.2f ' % med_dist) + text
        
    if not axis:
        axis = plt.gca()
            
    x = [positions[0], positions[0], positions[1], positions[1]]
    axis.plot(x, [y, y+h, y+h, y], lw=1.5, c=col)
    axis.text(np.mean(positions), y+h*1.4, text,
              ha='center', va='bottom', color=col)
    if adjust_y:
        axis.set_ylim(top=y+4*h)
        
def plot_with_shade(x, data, quantile = [0,1], color=None, marker = None):
    if not color:
        p = plt.plot(x,[np.ma.median(d) for d in data], marker=marker)
        plt.fill_between(x,[np.nanquantile(np.ma.filled(d, fill_value=np.nan),quantile[0])for d in data], 
                           [np.nanquantile(np.ma.filled(d, fill_value=np.nan),quantile[1]) for d in data],
                           alpha=0.5, color = p[0].get_color(), linewidth=0)       
    else:
        p = plt.plot(x,[np.ma.median(d) for d in data],color=color, marker=marker)
        plt.fill_between(x,[np.nanquantile(np.ma.filled(d, fill_value=np.nan),quantile[0])for d in data], 
                           [np.nanquantile(np.ma.filled(d, fill_value=np.nan),quantile[1]) for d in data],
                           alpha=0.5, color=color, linewidth=0)  
    return p[0]

def make_experiment_plots(tracking_folder : str):
    # PLOTS ----------------------------------------------------------------------------------------

    # make folder for plots
    plots_dir = os.path.join(current_experiment.path, tracking_folder, 'plots')
    try:
        os.mkdir(plots_dir)
    except OSError:
        pass
            
    # timesteps
    plt.figure(figsize=(9, 3))
    plt.plot(time_instants[:-1], time_steps)
    plt.title(f'Time step duration (expected={deltaT}s)')
    plt.xlabel('Time [s]')
    plt.gca().set_xlim([0, totalT])
    plt.gca().set_ylim(0)
    plt.ylabel('Time step [s]')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'timesteps.pdf'), bbox_inches = 'tight')
    plt.show()

    # number of agents
    plt.figure(figsize=(9,3))
    agents_number = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=1)
    plt.plot(time_instants,agents_number)
    plt.title(f'Number of detected agents over time (total={sum(lengths>min_traj_length)})')
    plt.xlabel('Time [s]')
    plt.gca().set_xlim([0, totalT])
    plt.gca().set_ylim(0)
    plt.ylabel('Count')
    plt.grid()
    DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    plt.savefig(os.path.join(plots_dir, 'number_of_agents.pdf'), bbox_inches = 'tight')
    plt.show()

    # length of trajectories histogram
    plt.title(f'Trajectories duration. Total={len(lengths)}, rejected={sum(lengths<min_traj_length)}')
    bins = [0,5,10,20,40,60,80,100, 120]
    plt.hist(lengths, bins)
    plt.axvline(min_traj_length, color='red')
    plt.xlabel('Time duration [s]')
    plt.xticks(bins)
    plt.gca().set_xlim([0, 120])
    plt.ylabel('Count')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'trajectories_length.pdf'), bbox_inches = 'tight')
    plt.show()


    # Inputs
    plt.figure(figsize=(9,6))
    plt.plot(time_instants,inputs[:,0], color='blue')
    plt.plot(time_instants,inputs[:,1], color='green')
    plt.plot(time_instants,inputs[:,2], color='red')
    plt.title('Inputs')
    plt.xlabel('Time [s]')
    plt.gca().set_xlim([0, totalT])
    plt.gca().set_ylim([0, 260])
    plt.ylabel('Brightness')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'inputs.pdf'), bbox_inches = 'tight')
    plt.show()

    # # directions histogram
    # plt.title('directions')
    # bins = np.linspace(-np.pi, np.pi, 9)
    # plt.hist(directions.flatten(), bins)
    # plt.xlabel('Direction [rad]')
    # plt.xticks(bins)
    # plt.gca().set_xlim([-np.pi, np.pi])
    # plt.ylabel('Count')
    # plt.grid()
    # plt.savefig(os.path.join(plots_dir, 'directions.pdf'), bbox_inches = 'tight')
    # plt.show()

    # # Average Speed, Acc, Angular Velocity, and Tumbling
    # plt.figure(figsize=(9,8))
    # plt.subplot(4, 1, 1)
    # #plt.plot(time_instants[:-1],np.ma.median(speeds,axis=1))
    # plt.plot(time_instants,np.ma.median(speeds_smooth,axis=1))
    # plt.fill_between(time_instants, np.min(speeds_smooth,axis=1), np.max(speeds_smooth,axis=1),alpha=0.5)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Speed [px/s]')
    # plt.gca().set_xlim([0, totalT])
    # plt.gca().set_ylim(0)
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)

    # plt.subplot(4, 1, 2)
    # plt.plot(time_instants,np.ma.median(acc_smooth,axis=1))
    # plt.fill_between(time_instants, np.min(acc_smooth,axis=1), np.max(acc_smooth,axis=1),alpha=0.5)
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Acc [px/s^2]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)

    # plt.subplot(4, 1, 3)
    # plt.plot(time_instants[:-1],np.ma.median(np.abs(ang_vel_smooth),axis=1))
    # plt.fill_between(time_instants[:-1], np.min(np.abs(ang_vel_smooth),axis=1), np.max(np.abs(ang_vel_smooth),axis=1),alpha=0.5)
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Ang Vel [rad/s]')
    # #plt.xlabel('Time [s]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)

    # plt.subplot(4, 1, 4)
    # plt.plot(time_instants,np.ma.mean(moving_average(tumbling2, 3),axis=1)*100)
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Tumbling [% of agents]')
    # plt.xlabel('Time [s]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    # plt.savefig(os.path.join(plots_dir, 'time_evolution.pdf'), bbox_inches = 'tight')
    # plt.show()

    # Time evolution of Average Speed and Angular Velocity
    plt.figure(figsize=(9,4))
    plt.subplot(2, 1, 1)
    #plt.plot(time_instants[:-1],np.ma.median(speeds,axis=1))
    plt.plot(time_instants,np.ma.median(speeds_smooth,axis=1))
    plt.fill_between(time_instants, np.min(speeds_smooth,axis=1), np.max(speeds_smooth,axis=1),alpha=0.5)
    #plt.xlabel('Time [s]')
    plt.ylabel('Speed [px/s]')
    plt.gca().set_xlim([0, totalT])
    plt.gca().set_ylim(0)
    plt.grid()
    DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)

    plt.subplot(2, 1, 2)
    plt.plot(time_instants[:-1],np.ma.median(np.abs(ang_vel_smooth),axis=1))
    plt.fill_between(time_instants[:-1], np.min(np.abs(ang_vel_smooth),axis=1), np.max(np.abs(ang_vel_smooth),axis=1),alpha=0.5)
    plt.gca().set_xlim([0, totalT])
    plt.ylabel('Ang. Vel. [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid()
    DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    plt.savefig(os.path.join(plots_dir, 'time_evolution.pdf'), bbox_inches = 'tight')
    plt.show()

    if len(speeds_on) >0:
        # Time evolution and boxplots of Average Speed and Angular Velocity
        fig = plt.figure(figsize=(9,4))
        ax_sp = fig.add_gridspec(bottom=0.55, right=0.75).subplots()
        ax_angv = fig.add_gridspec(top=0.45, right=0.75).subplots()
        ax_sp_box = ax_sp.inset_axes([1.05, 0, 0.25, 1], sharey=ax_sp)
        ax_sp_box.tick_params(axis="y", labelleft=False)
        ax_angv_box = ax_angv.inset_axes([1.05, 0, 0.25, 1], sharey=ax_angv)
        ax_angv_box.tick_params(axis="y", labelleft=False)
        #
        ax_sp.plot(time_instants,np.ma.median(speeds_smooth,axis=1))
        ax_sp.fill_between(time_instants, np.min(speeds_smooth,axis=1), np.max(speeds_smooth,axis=1),alpha=0.5)
        ax_sp.set_ylabel('Speed [px/s]')
        ax_sp.set_xlim([0, totalT])
        ax_sp.set_ylim([0, np.max(speeds_smooth)*0.9])
        ax_sp.grid()
        DOMEgraphics.highligth_inputs(inputs[:,0], time_instants, axis=ax_sp)
        data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on, axis=1), np.mean(speeds_off, axis=1)]))
        ax_sp_box.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        add_significance_bar(data=data_to_plot, axis=ax_sp_box)
        #
        ax_angv.plot(time_instants[:-1],np.ma.median(np.abs(ang_vel_smooth),axis=1))
        ax_angv.fill_between(time_instants[:-1], np.min(np.abs(ang_vel_smooth),axis=1), np.max(np.abs(ang_vel_smooth),axis=1),alpha=0.5)
        ax_angv.set_xlim([0, totalT])
        ax_angv.set_ylim([0, 2])
        ax_angv.set_ylabel('Ang. Vel. [rad/s]')
        plt.xlabel('Time [s]')
        ax_angv.grid()
        DOMEgraphics.highligth_inputs(inputs[:,0], time_instants, axis=ax_angv)
        data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on, axis=1), np.mean(ang_vel_off, axis=1)]))
        ax_angv_box.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        add_significance_bar(data=data_to_plot, axis=ax_angv_box)
        #
        plt.savefig(os.path.join(plots_dir, 'time_evolution_boxplots.pdf'), bbox_inches = 'tight')
        plt.show()


        # boxplots speed and ang vel of averages over the agents
        plt.figure(figsize=(3,4))
        plt.subplot(2, 1, 1)
        data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on, axis=1), np.mean(speeds_off, axis=1)]))
        plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        add_significance_bar(data=data_to_plot)
        plt.axhline(0, color='gray')
        plt.ylabel('Speed [px/s]')
        plt.title('Boxplots of averages over the agents')
        plt.subplot(2, 1, 2)
        data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on, axis=1), np.mean(ang_vel_off, axis=1)]))
        plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        add_significance_bar(data=data_to_plot)
        plt.axhline(0, color='gray')
        plt.ylabel('Ang Vel [rad/s]')
        plt.savefig(os.path.join(plots_dir, 'boxplots.pdf'), bbox_inches = 'tight')
        plt.show()


        # # boxplots of averages over time instants
        # plt.figure(figsize=(4,8))
        # plt.subplot(4, 1, 1)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on, axis=0), np.mean(speeds_off, axis=0), (np.mean(speeds_on, axis=0) - np.mean(speeds_off, axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Speed [px/s]')
        # plt.title('Boxplots')
        # plt.subplot(4, 1, 2)
        # data_to_plot = list(map(lambda X: [x for x in X if x],[np.mean(acc_on, axis=0), np.mean(acc_off, axis=0), (np.mean(acc_on, axis=0) - np.mean(acc_off, axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Acc [px/s^2]')
        # plt.subplot(4, 1, 3)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on, axis=0), np.mean(ang_vel_off, axis=0), (np.mean(ang_vel_on, axis=0) - np.mean(ang_vel_off, axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Ang Vel [rad/s]')
        # plt.subplot(4, 1, 4)
        # #data_to_plot =  [np.ma.mean(tumbling_on)*100, np.ma.mean(tumbling_off)*100, np.ma.mean(np.ma.mean(tumbling_on, axis=0) - np.ma.mean(tumbling_off, axis=0))*100]
        # #plt.bar([1, 2, 3], data_to_plot)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.ma.mean(tumbling_on, axis=0)*100, np.ma.mean(tumbling_off, axis=0)*100, (np.ma.mean(tumbling_on, axis=0) - np.ma.mean(tumbling_off, axis=0))*100]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Tumbling [% of frames]')
        # plt.savefig(os.path.join(plots_dir, 'boxplots.pdf'), bbox_inches = 'tight')
        # plt.show()
    
        # # boxplots speed and ang vel of averages over time instants
        # plt.figure(figsize=(3,4))
        # plt.subplot(2, 1, 1)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on, axis=0), np.mean(speeds_off, axis=0)]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Speed [px/s]')
        # plt.title('Boxplots of averages over time instants')
        # plt.subplot(2, 1, 2)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on, axis=0), np.mean(ang_vel_off, axis=0)]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Ang Vel [rad/s]')
        # plt.savefig(os.path.join(plots_dir, 'boxplots.pdf'), bbox_inches = 'tight')
        # plt.show()
    
        # # focused boxplots
        # plt.figure(figsize=(4,6))
        # plt.subplot(3, 1, 1)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(speeds_on[:5,:], axis=0), np.mean(speeds_off[-5:,:], axis=0), (np.mean(speeds_on[:5,:], axis=0) - np.mean(speeds_off[-5:,:], axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Speed [px/s]')
        # plt.title('Boxplots: just before and after the switch')
        # plt.subplot(3, 1, 2)
        # data_to_plot = list(map(lambda X: [x for x in X if x],[np.mean(acc_on[:5,:], axis=0), np.mean(acc_off[-5:,:], axis=0), (np.mean(acc_on[:5,:], axis=0) - np.mean(acc_off[-5:,:], axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Acc [px/s^2]')
        # plt.subplot(3, 1, 3)
        # data_to_plot = list(map(lambda X: [x for x in X if x], [np.mean(ang_vel_on[:5,:], axis=0), np.mean(ang_vel_off[-5:,:], axis=0), (np.mean(ang_vel_on[:5,:], axis=0) - np.mean(ang_vel_off[-5:,:], axis=0))]))
        # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF', 'Difference'])
        # plt.axhline(0, color='gray')
        # plt.ylabel('Ang Vel [rad/s]')
        # plt.savefig(os.path.join(plots_dir, 'focused_boxplots.pdf'), bbox_inches = 'tight')
        # plt.show()

    # histograms
    plt.figure(figsize=(4,6))
    plt.subplot(3, 1, 1)
    plt.title('Histograms')
    #bins=np.linspace(0, 40, round(40/5+1))
    my_histogram([np.mean(speeds_on, axis=0).compressed(), np.mean(speeds_off, axis=0).compressed()], normalize=True)
    plt.legend(labels=['Light ON', 'Light OFF'])
    plt.xlabel('Speed [px/s]')
    #plt.gca().set_ylim([0, 0.25])
    plt.ylabel('Agents')
    plt.grid()
    plt.subplot(3, 1, 2)
    #bins=np.linspace(0, 5, round(10+1))
    # my_histogram([np.mean(acc_on, axis=0).compressed(), np.mean(acc_off, axis=0).compressed()], normalize=True)
    plt.legend(labels=['Light ON', 'Light OFF'])
    plt.xlabel('Acc [px/s^2]')
    #plt.gca().set_ylim([0, 0.25])
    plt.ylabel('Agents')
    plt.grid()
    plt.subplot(3, 1, 3)
    #bins=np.linspace(0, 1, round(10+1))
    my_histogram([np.mean(ang_vel_on, axis=0).compressed(), np.mean(ang_vel_off, axis=0).compressed()], normalize=True)
    plt.legend(labels=['Light ON', 'Light OFF'])
    plt.xlabel('Ang Vel [rad/s]')
    #plt.gca().set_ylim([0, 0.25])
    plt.ylabel('Agents')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'histograms.pdf'), bbox_inches = 'tight')
    plt.show()


    # # focused histograms
    # plt.figure(figsize=(4,6))
    # plt.subplot(3, 1, 1)
    # plt.title('Histograms: just before and after the switch')
    # #bins=np.linspace(0, 40, round(40/5+1))
    # my_histogram([np.mean(speeds_on[:5,:], axis=0).compressed(), np.mean(speeds_off[-5:,:], axis=0).compressed()] , normalize=True)
    # plt.legend(labels=['Light ON', 'Light OFF'])
    # plt.xlabel('Speed [px/s]')
    # #plt.gca().set_ylim([0, 0.25])
    # plt.ylabel('Agents')
    # plt.grid()
    # plt.subplot(3, 1, 2)
    # #bins=np.linspace(0, 5, round(10+1))
    # my_histogram([np.mean(acc_on[:5,:], axis=0).compressed(), np.mean(acc_off[-5:,:], axis=0).compressed()] , normalize=True)
    # plt.legend(labels=['Light ON', 'Light OFF'])
    # plt.xlabel('Acc [px/s^2]')
    # #plt.gca().set_ylim([0, 0.25])
    # plt.ylabel('Agents')
    # plt.grid()
    # plt.subplot(3, 1, 3)
    # #bins=np.linspace(0, 1, round(10+1))
    # my_histogram([np.mean(ang_vel_on[:5,:], axis=0).compressed(), np.mean(ang_vel_off[-5:,:], axis=0).compressed()] , normalize=True)
    # plt.legend(labels=['Light ON', 'Light OFF'])
    # plt.xlabel('Ang Vel [rad/s]')
    # #plt.gca().set_ylim([0, 0.25])
    # plt.ylabel('Agents')
    # plt.grid()
    # plt.show()


    # # displacements autocorrelation
    # disp_acorr_df=pd.DataFrame(disp_acorr)
    # disp_acorr_df=disp_acorr_df.loc[:,0:20]
    # plt.figure()
    # plt.plot(np.linspace(0, 20, 21), disp_acorr_df.median(), marker='o')
    # plt.fill_between(np.linspace(0, 20, 21), disp_acorr_df.min(), disp_acorr_df.max(),alpha=0.5)
    # plt.title('displacements autocorrelation')
    # plt.grid()
    # plt.xlim([0, 20])
    # plt.show()

    # corrplot speed and ang vel (per agent)
    plt.figure(figsize=(9,6))
    agents_motion = pd.DataFrame(np.ma.array([np.ma.mean(speeds_smooth, axis=0), np.ma.mean(abs_ang_vel_smooth, axis=0), np.ma.std(speeds_smooth, axis=0), np.ma.std(abs_ang_vel_smooth, axis=0)]).T)
    agents_motion.columns = ['mean speed','mean ang vel','std speed','std ang vel']
    sns.pairplot(agents_motion);
    plt.savefig(os.path.join(plots_dir, 'corrplot_speed_angv.pdf'), bbox_inches = 'tight');
    plt.show();

    # scatter plot MEAN speed and MEAN ang velocity (per agent)
    plt.figure(figsize=(9,6))
    c = [np.ma.mean(speeds_smooth, axis=0)]                     #colored by speed
    #c = [np.array([i for i in range(speeds_smooth.shape[1])])] #colored by index
    scatter_hist([np.ma.mean(speeds_smooth, axis=0)], [np.ma.mean(abs_ang_vel_smooth, axis=0)], c, n_bins=10, cmap=DOMEgraphics.cropCmap('Blues', 0.25, 1.2))
    plt.xlabel('Mean Agent Speed [px/s]')
    plt.ylabel('Mean Agent Ang Vel [rad/s]')
    #plt.gca().set_xlim([0, 2.5])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_speed_angv_mean.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # scatter plot MEAN speed and STD speed (per agent)
    plt.figure(figsize=(9,6))
    c = [np.ma.mean(speeds_smooth, axis=0)]
    scatter_hist([np.ma.mean(speeds_smooth, axis=0)], [np.ma.std(speeds_smooth, axis=0)], c, n_bins=10, cmap=DOMEgraphics.cropCmap('Blues', 0.25, 1.2))
    plt.xlabel('Mean Agent Speed [px/s]')
    plt.ylabel('Std Agent Speed [px/s]')
    #plt.gca().set_xlim([0, 2.5])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_speed_mean_std.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # scatter plot MEAN ang velocity and STD ang velocity (per agent)
    plt.figure(figsize=(9,6))
    c = [np.ma.mean(speeds_smooth, axis=0)]
    scatter_hist([np.ma.mean(abs_ang_vel_smooth, axis=0)], [np.ma.std(abs_ang_vel_smooth, axis=0)], c, n_bins=10, cmap=DOMEgraphics.cropCmap('Blues', 0.25, 1.2))
    plt.xlabel('Mean Agent Ang Vel [rad/s]')
    plt.ylabel('Std Agent Ang Vel [rad/s]')
    #plt.gca().set_xlim([0, 2.5])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_angv_mean_std.pdf'), bbox_inches = 'tight')
    plt.show()

    # scatter plot STD speed and STD ang velocity (per agent)
    plt.figure(figsize=(9,6))
    c = [np.ma.mean(speeds_smooth, axis=0)]
    scatter_hist([np.ma.std(speeds_smooth, axis=0)], [np.ma.std(abs_ang_vel_smooth, axis=0)], c, n_bins=10, cmap=DOMEgraphics.cropCmap('Blues', 0.25, 1.2))
    plt.xlabel('Std Agents Speed [px/s]')
    plt.ylabel('Std Agents Ang Vel [rad/s]')
    #plt.gca().set_xlim([0, 2.5])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_speed_angv_std.pdf'), bbox_inches = 'tight')
    plt.show()

    # scatter plot speed variation and ang velocity variation
    fig=plt.figure(figsize=(9,6))
    speed_variation = [np.log10(speeds_smooth[:-1]/np.ma.mean(speeds_smooth, axis=0))]
    ang_vel_variation = [np.log10(abs_ang_vel_smooth/np.ma.mean(abs_ang_vel_smooth, axis=0))]
    c = np.ma.mean(speeds_smooth, axis=0)
    color = [np.tile(c, (len(time_instants)-1,1))]
    scatter_hist(speed_variation, ang_vel_variation, color, n_bins=50, cmap=DOMEgraphics.cropCmap('Blues', 0.25, 1.2))
    plt.xlabel('Speed / Agent mean speed [log10]')
    plt.ylabel('Ang Vel / Agent mean ang vel [log10]')
    # plt.gca().set_xlim([0, 20])
    # plt.gca().set_ylim([0, 20])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_speed_angv_variation_all.pdf'), bbox_inches = 'tight')
    plt.show()

    # # scatter plot speed and ang velocity - cluster wrt tumbling
    # plt.figure(figsize=(9,6))
    # x=split(np.ma.divide(speeds_smooth[:-1],np.ma.median(speeds_smooth, axis=0)), condition=tumbling2[:-1]<0.5)
    # y=split(abs_ang_vel_smooth, condition=tumbling2[:-1]<0.5)
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed / Agents median speed')
    # plt.ylabel('Ang Vel [rad/s]')
    # plt.gca().set_xlim([0, 2.5])
    # plt.legend(['running', 'tumbling'])
    # plt.grid()
    # plt.savefig(os.path.join(plots_dir, 'scatter_speed_angv_tumbling.pdf'), bbox_inches = 'tight')
    # plt.show()

    # scatter plot speed and ang velocity - cluster wrt light input
    plt.figure(figsize=(9,6))
    x=[np.ma.divide(speeds_on,np.ma.median(speeds_smooth, axis=0)), np.ma.divide(speeds_off[:-1],np.ma.median(speeds_smooth, axis=0))]
    y=[ang_vel_on, ang_vel_off]
    scatter_hist(x, y, n_bins=20)
    plt.xlabel('Speed / Agents median speed')
    plt.ylabel('Ang Vel [rad/s]')
    plt.gca().set_xlim([0, 2.5])
    plt.legend(['Light ON', 'Light OFF'])
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'scatter_speed_angv_light.pdf'), bbox_inches = 'tight')
    plt.show()

    # # scatter plot speed and lag1 similarity
    # plt.figure(figsize=(9,6))
    # x=[np.ma.divide(speeds_smooth,np.ma.median(speeds_smooth, axis=0))]
    # y=[lag1_similarity]
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed / Agents median speed')
    # plt.ylabel('Lag 1 similarity')
    # plt.gca().set_xlim([0, 2.5])
    # plt.grid()
    # plt.savefig(os.path.join(plots_dir, 'scatter_speed_lag1.pdf'), bbox_inches = 'tight')
    # plt.show()

    # # scatter plot speed and lag1 similarity - cluster wrt tumbling
    # plt.figure(figsize=(9,6))
    # x=split(np.ma.divide(speeds_smooth,np.ma.median(speeds_smooth, axis=0)), condition=tumbling2<0.5)
    # y=split(lag1_similarity, condition=tumbling2<0.5)
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed / Agents median speed')
    # plt.ylabel('Lag 1 similarity')
    # plt.gca().set_xlim([0, 2.5])
    # plt.legend(['running', 'tumbling'])
    # plt.grid()
    # plt.savefig(os.path.join(plots_dir, 'scatter_speed_lag1_tumbling.pdf'), bbox_inches = 'tight')
    # plt.show()

    # # scatter plot speed and lag1 similarity - cluster wrt light input
    # plt.figure(figsize=(9,6))
    # x=[np.ma.divide(speeds_on,np.ma.median(speeds_smooth, axis=0)), np.ma.divide(speeds_off,np.ma.median(speeds_smooth, axis=0))]
    # y=[lag1_similarity_on, lag1_similarity_off]
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed / Agents median speed')
    # plt.ylabel('Lag 1 similarity')
    # plt.gca().set_xlim([0, 2.5])
    # plt.legend(['Light ON', 'Light OFF'])
    # plt.grid()
    # plt.savefig(os.path.join(plots_dir, 'scatter_speed_lag1_light.pdf'), bbox_inches = 'tight')
    # plt.show()

    # # heatmap input - tumbling
    # plt.figure(figsize=(4,4))
    # x=np.array([[np.ma.sum(tumbling_on),
    #     np.ma.sum(tumbling_off)], 
    #    [np.ma.sum(-tumbling_on+1), 
    #     np.ma.sum(-tumbling_off+1)]])
    # x=(x.T/np.ma.sum(x, axis=1)).T
    # sns.heatmap(x, xticklabels=['Light ON','Light OFF'], yticklabels=['Tumbling','Running'], 
    #             annot=True, cbar=False, vmin=0.25, vmax=0.75, cmap="gray", linewidths=0.2)
    # plt.savefig(os.path.join(plots_dir, 'tumbling_light.pdf'), bbox_inches = 'tight')


    # Select one agent ---------------------------------------------------------------------------------
    agent=np.argmax(lengths)
    agent=random.choice(np.arange(len(lengths))[lengths >= min_traj_length])
    #agent= 102 #145 #127 #40 #109

    # # Speed and Acceleration of one agent
    # plt.figure(figsize=(9,6))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_instants,speeds_smooth[:,agent])
    # #plt.plot(time_instants,speeds[:,agent], '--')
    # plt.title('Movement of agent '+str(agent))
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Speed [px/s]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    # #DOMEgraphics.highligth_inputs(tumbling[:,agent].astype(float), 'green')
    # DOMEgraphics.highligth_inputs(tumbling2[:,agent].astype(float), time_instants,'yellow')


    # plt.subplot(2, 1, 2)
    # plt.plot(time_instants,acc_smooth[:,agent])
    # #plt.plot(time_instants,np.abs(acc[:,agent]),'--')
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Abs Acc [px/s^2]')
    # plt.xlabel('Time [s]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    # #DOMEgraphics.highligth_inputs(tumbling[:,agent].astype(float), 'green')
    # DOMEgraphics.highligth_inputs(tumbling2[:,agent].astype(float), time_instants,'yellow')
    # plt.show()

    # # Direction and Angular Velocity of one agent
    # plt.figure(figsize=(9,6))
    # plt.subplot(2, 1, 1)
    # #plt.plot(time_instants[1:],directions[:,agent])
    # plt.plot(time_instants[1:],directions_reg_smooth[:,agent])
    # #plt.plot(time_instants[1:],directions_reg[:,agent],'--')
    # plt.title('Movement of agent '+str(agent))
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Direction [rad]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    # #DOMEgraphics.highligth_inputs(tumbling[:,agent].astype(float), 'green')
    # DOMEgraphics.highligth_inputs(tumbling2[:,agent].astype(float), time_instants,'yellow')

    # # plt.subplot(3, 1, 2)
    # # plt.plot(time_instants[:-1],directions[:,agent])
    # # plt.plot(time_instants[:-1],directions_smooth[:,agent])
    # # plt.title('Movement of agent '+str(agent))
    # # plt.gca().set_xlim([0, totalT])
    # # plt.gca().set_ylim([-np.pi, np.pi])
    # # plt.ylabel('Direction [rad]')
    # # plt.yticks(np.linspace(-np.pi, np.pi, 5))
    # # plt.grid()
    # # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)

    # plt.subplot(2, 1, 2)
    # plt.plot(time_instants[1:],np.abs(ang_vel_smooth[:,agent]))
    # #plt.plot(time_instants[1:],np.abs(ang_vel[:,agent]),'--')
    # plt.gca().set_xlim([0, totalT])
    # plt.ylabel('Abs Angular Vel [rad/s]')
    # plt.xlabel('Time [s]')
    # plt.grid()
    # DOMEgraphics.highligth_inputs(inputs[:,0], time_instants)
    # #DOMEgraphics.highligth_inputs(tumbling[:,agent].astype(float), 'green')
    # DOMEgraphics.highligth_inputs(tumbling2[:,agent].astype(float), time_instants,'yellow')
    # plt.show()

    # # Barplots of one agent
    # plt.figure(figsize=(4,8))
    # plt.subplot(4, 1, 1)
    # data_to_plot = list(map(lambda X: [x for x in X if x], [speeds_on[:,agent], speeds_off[:,agent]]))
    # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
    # plt.ylabel('Speed [px/s]')
    # plt.title('Average values of agent '+ str(agent))
    # plt.subplot(4, 1, 2)
    # data_to_plot = list(map(lambda X: [x for x in X if x], [acc_on[:,agent], acc_off[:,agent]]))
    # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
    # plt.ylabel('Acc [px/s^2]')
    # plt.subplot(4, 1, 3)
    # data_to_plot = list(map(lambda X: [x for x in X if x], [ang_vel_on[:,agent], ang_vel_off[:,agent]]))
    # plt.boxplot(data_to_plot,labels=['Light ON', 'Light OFF'])
    # plt.ylabel('Ang Vel [rad/s]')
    # plt.subplot(4, 1, 4)
    # data_to_plot =  [np.mean(tumbling_on[:,agent])*100, np.mean(tumbling_off[:,agent])*100]
    # plt.bar([1, 2], data_to_plot)
    # plt.xticks([1, 2],labels=['Light ON', 'Light OFF'])
    # plt.ylabel('Tumbling [% of frames]')
    # plt.show()


    # disp_acorr=vector_autocorrelation(displacements[:,agent,:])
    # disp_corr=vector_correlation(displacements[:,agent,:], displacements[:,agent,:])
    # disp_similarity = vector_similarity(displacements[:,agent,:], displacements[:,agent,:])

    # signal_similarity=np.max(correlation(disp_corr, disp_acorr),axis=1)
    # signal_difference=np.zeros([len(disp_corr)])*np.nan
    # for i in range(round(len(disp_corr)-5)):
    #     signal_difference[i]=np.linalg.norm(disp_corr[i,i:i+5]- disp_acorr[0:5])/len(disp_acorr[0:5])


    # # displacements autocorrelation
    # plt.figure()
    # for i in range(round(len(disp_corr)/2)):
    #     plt.plot(disp_corr[i,i:])
    #     plt.plot(disp_acorr, marker='o')
    #     plt.title('displacements autocorrelation of agent '+ str(agent)+ 't='+str(i))
    #     plt.xlabel('lag')
    #     plt.grid()
    #     plt.show()

    # # displacements self similarity
    # plt.figure()
    # plt.plot(vector_auto_similarity(displacements[:,agent,:]))
    # plt.title('displacements self similarity of agent '+ str(agent))
    # plt.xlabel('lag')
    # plt.grid()
    # plt.show()

    # # lag 1 displacements similarity
    # plt.figure()
    # plt.plot(time_instants,lag1_similarity[:,agent])
    # plt.axhline(lag1_similarity_thresh,color='gray')
    # plt.title('lag 1 displacements similarity of agent '+ str(agent))
    # plt.xlabel('time [s]')
    # plt.ylim([0, 1.1])
    # #plt.xlim([0, totalT])
    # plt.grid()
    # #DOMEgraphics.highligth_inputs(tumbling[:,agent].astype(float), 'green')
    # DOMEgraphics.highligth_inputs(tumbling2[:,agent].astype(float), time_instants,'yellow')
    # plt.show()

    # # displacements signal similarity
    # plt.figure()
    # plt.plot(np.diff(signal_similarity))
    # plt.plot(signal_similarity)
    # plt.plot(signal_difference)
    # plt.title('displacements signal similarity of agent '+ str(agent))
    # plt.xlabel('time')
    # plt.grid()
    # plt.show()


    # # scatter plot speed and ang velocity of one agent clustered on Tumbling
    # plt.figure(figsize=(9,6))
    # x=split(speeds_smooth[:-1, agent], condition=tumbling2[:-1, agent]<0.5)
    # y=split(np.ma.abs(ang_vel_smooth[:,agent]), condition=tumbling2[:-1, agent]<0.5)
    # scatter_hist(x, y)
    # plt.xlabel('Speed [px/s]')
    # plt.ylabel('Ang Vel [rad/s]')
    # #plt.gca().set_ylim([0, 0.25])
    # plt.legend(['running', 'tumbling'])
    # plt.title('Agent '+ str(agent))
    # plt.grid()
    # plt.show()

    # scatter plot speed and ang velocity of one agent
    fig = plt.figure(figsize=(9,6))
    x=[speeds_smooth[:-1, agent]]
    y=[np.ma.abs(ang_vel_smooth[:,agent])]
    scatter_hist(x, y)
    plt.xlabel('Speed [px/s]')
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_ylim([0, 0.25])
    fig.suptitle('Agent '+ str(agent))
    plt.grid()
    plt.show()

    # # scatter plot speed and lag1 similarity clustering on Tumbling
    # plt.figure(figsize=(9,6))
    # x=split(speeds_smooth[:, agent], condition=tumbling2[:, agent]<0.5)
    # y=split(lag1_similarity[:,agent], condition=tumbling2[:, agent]<0.5)
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed [px/s]')
    # plt.ylabel('Lag 1 similarity')
    # plt.title('Agent '+ str(agent))
    # plt.grid()
    # plt.show()


    # # scatter plot speed and lag1 similarity
    # plt.figure(figsize=(9,6))
    # x=[speeds_smooth[:, agent]]
    # y=[lag1_similarity[:,agent]]
    # scatter_hist(x, y, n_bins=20)
    # plt.xlabel('Speed [px/s]')
    # plt.ylabel('Lag 1 similarity')
    # plt.title('Agent '+ str(agent))
    # plt.grid()
    # plt.show()

    # v(k+1) vs v(k) plot 
    fig=plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    x=[speeds_smooth[:-1, agent]]
    y=[speeds_smooth[1:, agent]]
    plt.scatter(x, y)
    plt.xlabel('Speed at time k [px/s]')
    plt.ylabel('Speed at time k+1 [px/s]')
    plt.grid()

    plt.subplot(1, 2, 2)
    x=[ang_vel_smooth[:-1, agent]]
    y=[ang_vel_smooth[1:, agent]]
    plt.scatter(x, y)
    plt.xlabel('Ang Vel at time k [rad/s]')
    plt.ylabel('Ang Vel at time k+1 [rad/s]')
    plt.grid()
    fig.suptitle('Agent '+ str(agent))
    plt.show()


    # plot trajectory of one agent
    last_index = inactivity.shape[0] - (inactivity[:, agent]<=0)[::-1].argmax(0) - 1
    img = current_experiment.get_img_at_time(last_index*deltaT)
    DOMEgraphics.draw_trajectories(interp_positions[:,agent:agent+1,:], [], inactivity[:,agent:agent+1], img, "trajectory of agent " +str(agent));
    # #tumbling_pos = interp_positions[:-1,agent,:][tumbling[:,agent]>0]
    # tumbling_pos2 = interp_positions[:,agent,:][tumbling2[:,agent]>0]
    # #plt.scatter(tumbling_pos[:,0], tumbling_pos[:,1], color='green' )
    # plt.scatter(tumbling_pos2[:,0], tumbling_pos2[:,1], color='yellow' )
    plt.show()


    
def analyse_experiment(experiment_name : str, tracking_folder : str ='last' ):
    global positions, inactivity, inputs, time_instants, time_steps, deltaT, totalT
    global current_experiment, interp_positions, speeds_smooth, acc_smooth
    global ang_vel_smooth, speeds_on, speeds_off, ang_vel_on, ang_vel_off, lengths, abs_ang_vel_smooth
    
    ## LOAD EXPERIMENT DATA
    if type(experiment_name) is str:
        current_experiment = DOMEexp.open_experiment(experiment_name, experiments_directory)  
    else:
        current_experiment = experiment_name
    
    totalT = current_experiment.get_totalT()  
    deltaT = float(current_experiment.get_deltaT())  
    with current_experiment.get_data('data.npz') as data:
        activation_times = data['activation_times']

    time_steps = np.diff(activation_times)
    time_instants = np.arange(stop=totalT+deltaT, step=deltaT)
    patterns = [current_experiment.get_pattern_at_time(t) for t in time_instants]

    # inputs
    inputs = np.mean(np.mean(patterns, axis=1), axis=1)
    
    ## LOAD TRACKING DATA
    if tracking_folder=='last':
        tracking_folder = current_experiment.get_last_tracking()
    positions, inactivity, *_ = DOMEtracker.load_tracking(tracking_folder, current_experiment)
    # number_of_agents = positions.shape[1]
    
    ## PERFORM TRAJECTORIES ANALYSIS (IF NOT EXISTING) AND LOAD DATA
    analysed_data_path = os.path.join(current_experiment.path, tracking_folder, "analysed_data.npz")

    if not os.path.isfile(analysed_data_path):
        print("Analysing trajectories...")
        analyse_trajectories(current_experiment, tracking_folder)
    else:
        print("Loading existing analysis data...")
        
    with current_experiment.get_data(analysed_data_path) as analysed_data:
        interp_positions=analysed_data["interp_positions"]
        interp_positions = np.ma.array(interp_positions, mask=np.isnan(interp_positions))
        speeds_smooth=analysed_data["speeds_smooth"]
        speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))
        acc_smooth=analysed_data["acc_smooth"]
        acc_smooth = np.ma.array(acc_smooth, mask=np.isnan(acc_smooth))
        ang_vel_smooth=analysed_data["ang_vel_smooth"]
        ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
        speeds_on=analysed_data["speeds_on"]
        speeds_on = np.ma.array(speeds_on, mask=np.isnan(speeds_on))
        speeds_off=analysed_data["speeds_off"]
        speeds_off = np.ma.array(speeds_off, mask=np.isnan(speeds_off))
        ang_vel_on=analysed_data["ang_vel_on"]
        ang_vel_on = np.ma.array(ang_vel_on, mask=np.isnan(ang_vel_on))
        ang_vel_off=analysed_data["ang_vel_off"]
        ang_vel_off = np.ma.array(ang_vel_off, mask=np.isnan(ang_vel_off))    
            
    # other quantities
    deltaT = float(current_experiment.get_deltaT())
    lengths = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=0) * deltaT
    abs_ang_vel_smooth = np.ma.abs(ang_vel_smooth)
    
    make_experiment_plots(tracking_folder)

def analyse_trajectories(experiment : [str, DOMEexp.ExperimentManager], tracking_folder : str ='last' ):
    ## LOAD EXPERIMENT DATA
    if type(experiment) is str:
        current_experiment= DOMEexp.open_experiment(experiment, experiments_directory)  
    else:
        current_experiment = experiment
    
    totalT = current_experiment.get_totalT()  
    deltaT = float(current_experiment.get_deltaT())  
    with current_experiment.get_data('data.npz') as data:
        activation_times = data['activation_times']

    time_steps = np.diff(activation_times)
    time_instants = np.arange(stop=totalT+deltaT, step=deltaT)
    patterns = [current_experiment.get_pattern_at_time(t) for t in time_instants]

    # inputs
    inputs = np.mean(np.mean(patterns, axis=1), axis=1)

    ## LOAD TRACKING DATA
    if tracking_folder=='last':
        tracking_folder = current_experiment.get_last_tracking()
    positions, inactivity, *_ = DOMEtracker.load_tracking(tracking_folder, current_experiment)
    number_of_agents = positions.shape[1]

    # plot trajectories
    img = current_experiment.get_img_at_time(totalT)
    DOMEgraphics.draw_trajectories(positions, inactivity=inactivity, img=img, title="trajectories", max_inactivity=0)


    # replace estimated positions with interpolated ones and apply uniform time sampling
    positions[inactivity!=0]=np.nan
    interp_positions = DOMEtracker.interpolate_positions(positions, activation_times, time_instants)
    #interp_positions = DOMEtracker.interpolate_positions(positions)
    DOMEgraphics.draw_trajectories(interp_positions, inactivity=inactivity, img=img, title="interpolated trajectories", max_inactivity=0)

    # smooth trajectories
    #interp_positions = np.ma.array(interp_positions, mask=np.isnan(interp_positions))
    interp_positions[:,:,0] = moving_average(interp_positions[:,:,0],3)
    interp_positions[:,:,1] = moving_average(interp_positions[:,:,1],3)
    # interp_positions[:,:,0] = moving_average(interp_positions[:,:,0],3)
    # interp_positions[:,:,1] = moving_average(interp_positions[:,:,1],3)
    DOMEgraphics.draw_trajectories(interp_positions, inactivity=inactivity, img=img, title="smoothed trajectories", max_inactivity=0)


    # length of trajectories
    lengths = np.count_nonzero(~np.isnan(interp_positions[:,:,0]), axis=0) * deltaT

    # discard short trajectories
    interp_positions[:,lengths<min_traj_length,:]= np.nan

    # velocities
    #displacements = np.gradient(interp_positions, axis=0)        # [px/frame]
    velocities = np.gradient(interp_positions, deltaT, axis=0)    # [px/s]

    # speed [px/s]
    speeds = np.linalg.norm(velocities, axis=2) 
    speeds = np.ma.array(speeds, mask=np.isnan(speeds))

    speeds_smooth = moving_average(speeds, 3)
    speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))

    # accelearation [px/s^2]
    acc = np.gradient(speeds_smooth, axis=0)                                
    acc = np.ma.array(acc, mask=np.isnan(acc))
    acc_smooth = moving_average(acc, 3)
    acc_smooth = np.ma.array(acc_smooth, mask=np.isnan(acc_smooth))

    # reject outliers
    outliers_speed=detect_outliers(speeds_smooth, m=variance_thresh, side='top')
    outliers_acc=detect_outliers(acc_smooth, m=variance_thresh, side='top')
    outliers = outliers_speed * outliers_acc
    for i in range(number_of_agents):
        if np.ma.max(outliers[:,i]):
            print('Agent '+str(i)+' is an outlier at time ' + str(np.argmax(outliers[:,i])*deltaT)+ 
                  '. Consider removing it with remove_agent(id).')

    # directions
    norm_disp = np.divide(velocities,np.stack([speeds,speeds], axis=2)+0.001)
    norm_disp = np.ma.array(norm_disp, mask=np.isnan(norm_disp))
    directions=np.arctan2(norm_disp[:,:,1],norm_disp[:,:,0])

    # compue angular velocity [rad/s]
    ang_vel = angle_diff(directions[1:,:], directions[:-1,:])
    ang_vel = np.ma.array(ang_vel, mask=np.isnan(ang_vel))

    # inergrate angular velocity to obtain continous direction
    starting_dir = np.zeros([1, directions.shape[1]])
    for i in range(directions.shape[1]):
        # starting_idx = np.ma.flatnotmasked_edges(directions[:,i])
        try:
            starting_dir[0,i]= directions[np.ma.flatnotmasked_edges(directions[:,i])[0], i] 
        except:
            pass
    directions_reg = starting_dir + np.cumsum(ang_vel, axis=0)
    directions_reg[directions_reg.mask==1] = np.nan
    directions_reg_smooth = moving_average(directions_reg, 3)

    # differentiate continous direction to obtain smooth ang vel [rad/s]
    ang_vel_smooth = np.gradient(directions_reg_smooth, deltaT, axis=0)
    #ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
    ang_vel_smooth = moving_average(ang_vel_smooth, 3)
    ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
    # abs_ang_vel_smooth = np.ma.abs(ang_vel_smooth)

    # autocorrelation of velocities
    # disp_acorr=[]
    # disp_corr=[]
    lag1_similarity=np.zeros(speeds_smooth.shape)
    for agent in range(number_of_agents):
        # disp_acorr.append(vector_autocorrelation(displacements[:,agent,:]))
        # disp_corr.append(vector_correlation(displacements[:,agent,:], displacements[:,agent,:]))
        lag1_similarity[:,agent]= lag_auto_similarity(velocities[:,agent,:])
    lag1_similarity = np.ma.array(lag1_similarity, mask=np.isnan(lag1_similarity))

    # detect tumbling
    var_thresh = 1
    lag1_similarity_thresh = 0.75
    tumbling=np.zeros(ang_vel_smooth.shape)
    tumbling2=np.zeros(speeds_smooth.shape)
    for agent in range(number_of_agents):
        tumbling[:,agent] = detect_tumbling(speeds_smooth[:,agent], ang_vel_smooth[:,agent], var_thresh)
        tumbling2[:,agent] = detect_outliers(lag1_similarity[:,agent], 1.5, 'bottom') * (lag1_similarity[:,agent] < lag1_similarity_thresh)
    tumbling = np.ma.array(tumbling, mask=np.isnan(ang_vel_smooth))
    tumbling2 = np.ma.array(tumbling2, mask=np.isnan(speeds_smooth))
    # tumbling=np.ma.filled(tumbling, np.nan)
    # tumbling2=np.ma.filled(tumbling2, np.nan)

    # inputs
    inputs = np.mean(np.mean(patterns, axis=1), axis=1)

    # values for different inputs
    [speeds_on, speeds_off] = split(speeds_smooth, condition=inputs[:,0]>=50)
    [acc_on, acc_off] = split(acc_smooth, condition=inputs[:,0]>=50)
    [ang_vel_on, ang_vel_off] = split(np.abs(ang_vel_smooth), condition=inputs[:-1,0]>=50)
    [tumbling_on, tumbling_off] = split(tumbling2, condition=inputs[:,0]>=50)
    [lag1_similarity_on, lag1_similarity_off] = split(lag1_similarity, condition=inputs[:,0]>=50)

    # T-test
    #scipy.stats.ttest_ind(np.mean(speeds_on, axis=1), np.mean(speeds_off, axis=1))
    
    # Save analysis data
    current_experiment.save_data(os.path.join(tracking_folder, 'analysed_data'), force=True, time_steps=time_steps, 
                                  interp_positions=interp_positions, speeds_smooth=speeds_smooth, acc_smooth=acc_smooth,
                                  ang_vel_smooth=ang_vel_smooth, speeds_on=speeds_on, speeds_off=speeds_off,
                                  ang_vel_on=ang_vel_on, ang_vel_off=ang_vel_off)
    
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'speeds_smooth'), force=True, data=speeds_smooth)
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'ang_vel_smooth'), force=True, data=ang_vel_smooth)
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'speeds_on'), force=True, data=speeds_on)
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'speeds_off'), force=True, data=speeds_off)
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'ang_vel_on'), force=True, data=ang_vel_on)
    current_experiment.save_data_txt(os.path.join(tracking_folder, 'ang_vel_off'), force=True, data=ang_vel_off)
    
    print(f'Analysis data saved as {os.path.join(current_experiment.path, tracking_folder, "analysed_data.npz")}')

def experiments_comparison(experiment_names : List, tracking_folders : [str, List] ='last' ):
    number_of_exp = len(experiment_names)
    
    if type(tracking_folders) is str:
        tracking_folders = [tracking_folders] * number_of_exp
    
    speeds_on=[]
    speeds_off=[]
    ang_vel_on=[]
    ang_vel_off=[]
    
    speeds_all = []
    ang_vel_all = []
    speeds_all_norm=[]
    ang_vel_all_norm=[]
    speeds_means=[]
    ang_vel_means=[]
    speeds_means_norm=[]
    ang_vel_means_norm=[]
    
    #load data from the experiments
    for (experiment_name, tracking_folder) in zip(experiment_names,tracking_folders):
        current_experiment= DOMEexp.open_experiment(experiment_name, experiments_directory)  
        if tracking_folder=='last':
            tracking_folder = current_experiment.get_last_tracking()
        print(f'{tracking_folder}')
        
        analysed_data_path = os.path.join(current_experiment.path, tracking_folder, "analysed_data.npz")
        if not os.path.isfile(analysed_data_path):
            print("Analysing trajectories...")
            analyse_trajectories(current_experiment, tracking_folder)
            
        with current_experiment.get_data(analysed_data_path) as analysed_data:
            #time_steps=analysed_data["time_steps"]
            #speeds_smooth=analysed_data["speeds_smooth"]
            #speeds_smooth = np.ma.array(speeds_smooth, mask=np.isnan(speeds_smooth))
            #ang_vel_smooth=analysed_data["ang_vel_smooth"]
            #ang_vel_smooth = np.ma.array(ang_vel_smooth, mask=np.isnan(ang_vel_smooth))
            speeds_on.append(analysed_data["speeds_on"])
            speeds_on[-1] = np.ma.array(speeds_on[-1], mask=np.isnan(speeds_on[-1]))
            speeds_off.append(analysed_data["speeds_off"])
            speeds_off[-1] = np.ma.array(speeds_off[-1], mask=np.isnan(speeds_off[-1]))
            ang_vel_on.append(analysed_data["ang_vel_on"])
            ang_vel_on[-1] = np.ma.array(ang_vel_on[-1], mask=np.isnan(ang_vel_on[-1]))
            ang_vel_off.append(analysed_data["ang_vel_off"])
            ang_vel_off[-1] = np.ma.array(ang_vel_off[-1], mask=np.isnan(ang_vel_off[-1]))    
       
    for i in range(number_of_exp):
        if len(speeds_on[i]) > 0:
            speeds_all.append([speeds_off[i].flatten(), speeds_on[i].flatten()])
            ang_vel_all.append([ang_vel_off[i].flatten(), ang_vel_on[i].flatten()])
            speeds_all_norm.append([speeds_all[i][0]/np.ma.median(speeds_all[i][0]), speeds_all[i][1]/np.ma.median(speeds_all[i][0])])
            ang_vel_all_norm.append([ang_vel_all[i][0]/np.ma.median(ang_vel_all[i][0]), ang_vel_all[i][1]/np.ma.median(ang_vel_all[i][0])])
        
            speeds_means.append([np.mean(speeds_off[i], axis=1), np.mean(speeds_on[i], axis=1)])
            ang_vel_means.append([np.mean(ang_vel_off[i], axis=1) ,np.mean(ang_vel_on[i], axis=1)])
            speeds_means_norm.append([speeds_means[i][0]/np.ma.median(speeds_means[i][0]), speeds_means[i][1]/np.ma.median(speeds_means[i][0])])
            ang_vel_means_norm.append([ang_vel_means[i][0]/np.ma.median(ang_vel_means[i][0]), ang_vel_means[i][1]/np.ma.median(ang_vel_means[i][0])])
        else:
            speeds_all.append([speeds_off[i].flatten()])
            ang_vel_all.append([ang_vel_off[i].flatten()])
            speeds_all_norm.append([speeds_all[i][0]/np.ma.median(speeds_all[i][0])])
            ang_vel_all_norm.append([ang_vel_all[i][0]/np.ma.median(ang_vel_all[i][0])])
        
            speeds_means.append([np.mean(speeds_off[i], axis=1)])
            ang_vel_means.append([np.mean(ang_vel_off[i], axis=1)])
            speeds_means_norm.append([speeds_means[i][0]/np.ma.median(speeds_means[i][0])])
            ang_vel_means_norm.append([ang_vel_means[i][0]/np.ma.median(ang_vel_means[i][0])])
        
    # PLOTS
    plots_dir = '/Volumes/DOMEPEN/Experiments/comparisons'
    with open(os.path.join(plots_dir, 'comparison_info.txt'), 'w') as file:
        file.write(f'Experiments Comparison \nExperiments: {experiment_names}\nTracking: {tracking_folders}')
    
    # comparison speed and ang vel of all values, across comparison
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_all, compare='across', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('All values')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_all, compare='across', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_all_values_across.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of all values, within comparison
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_all, compare='within', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('All values')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_all, compare='within', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_all_values_within.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of all values, normalised over OFF median
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_all_norm, compare='within', whis = 3)
    #plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('All values normalized on OFF median value')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_all_norm, compare='within', whis = 3)
    #plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_all_values_norm.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of all values, aggregate
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot([np.ma.concatenate([d[0] for d in speeds_all]), np.ma.concatenate([d[1] for d in speeds_all])], compare='within', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('Aggregate all values')
    plt.subplot(2, 1, 2)
    my_boxplot([np.ma.concatenate([d[0] for d in ang_vel_all]), np.ma.concatenate([d[1] for d in ang_vel_all])], compare='within', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_all_values_aggregate.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, across comparison
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_means, compare='across', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('Averages over agents')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_means, compare='across', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_mean_values_across.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, within comparison
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_means, compare='within', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('Averages over agents')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_means, compare='within', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_mean_values_within.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, normalised over OFF median
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot(speeds_means_norm, compare='within', whis = 3)
    #plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('Averages over agents normalized on median OFF value')
    plt.subplot(2, 1, 2)
    my_boxplot(ang_vel_means_norm, compare='within', whis = 3)
    #plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_mean_values_norm.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, aggregate
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    my_boxplot([np.ma.concatenate([d[0] for d in speeds_means]), np.ma.concatenate([d[1] for d in speeds_means])], compare='within', whis = 3)
    plt.ylabel('Speed [px/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.title('Aggregate averages over agents')
    plt.subplot(2, 1, 2)
    my_boxplot([np.ma.concatenate([d[0] for d in ang_vel_means]), np.ma.concatenate([d[1] for d in ang_vel_means])], compare='within', whis = 3)
    plt.ylabel('Ang Vel [rad/s]')
    #plt.gca().set_xticklabels(labels=['Light OFF', 'Light ON'])
    plt.savefig(os.path.join(plots_dir, 'comparison_mean_values_aggregate.pdf'), bbox_inches = 'tight')
    plt.show()
    

def scenarios_comparison(experiment_names : List, labels = None, xvalues = None, tracking_folders : [str, List] ='last' ):
    number_of_scenarios = len(experiment_names)
    number_of_exp = len(experiment_names[0])
    
    if not any(xvalues): xvalues=np.arange(number_of_scenarios)
    
    if type(tracking_folders) is str:
        tracking_folders = [[tracking_folders] * number_of_exp] * number_of_scenarios
    
    speeds_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_ref=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_ref=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_on_all=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_off_all=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_on_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_off_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_on_all=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_off_all=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_on_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_off_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    
    speeds_mean_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_mean_ref=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_ref=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_mean_on_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_mean_off_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_on_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_off_norm=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_mean_on_norm2=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    speeds_mean_off_norm2=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_on_norm2=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    ang_vel_mean_off_norm2=[list(range(number_of_exp)) for i in range(number_of_scenarios)]
    
    
    #load data from all the experiments
    for scen in range(number_of_scenarios):
        for exp in range(number_of_exp):
            current_experiment= DOMEexp.open_experiment(experiment_names[scen][exp], experiments_directory)  
            tracking_folder = tracking_folders[scen][exp]
            if tracking_folder=='last':
                tracking_folder = current_experiment.get_last_tracking()
            print(f'{tracking_folder}')
            
            analysed_data_path = os.path.join(current_experiment.path, tracking_folder, "analysed_data.npz")
            if not os.path.isfile(analysed_data_path):
                print("Analysing trajectories...")
                analyse_trajectories(current_experiment, tracking_folder)
                
            with current_experiment.get_data(analysed_data_path) as analysed_data:
                speeds = analysed_data["speeds_smooth"]
                speeds = np.ma.array(speeds, mask=np.isnan(speeds))
                ang_vel = np.ma.abs(analysed_data["ang_vel_smooth"])
                ang_vel = np.ma.array(ang_vel, mask=np.isnan(ang_vel))
                speeds_on = analysed_data["speeds_on"]
                speeds_on = np.ma.array(speeds_on, mask=np.isnan(speeds_on))
                speeds_off = analysed_data["speeds_off"]
                speeds_off = np.ma.array(speeds_off, mask=np.isnan(speeds_off))
                ang_vel_on = np.ma.abs(analysed_data["ang_vel_on"])
                ang_vel_on = np.ma.array(ang_vel_on, mask=np.isnan(ang_vel_on))
                ang_vel_off = np.ma.abs(analysed_data["ang_vel_off"])
                ang_vel_off = np.ma.array(ang_vel_off, mask=np.isnan(ang_vel_off))
                
                # all values
                speeds_ref[scen][exp]   = speeds[0:20].flatten()
                speeds_norm[scen][exp]  = speeds[20:].flatten()
                speeds_norm[scen][exp]  = speeds_norm[scen][exp] / np.ma.median(speeds_ref[scen][exp])
        
                speeds_on_all[scen][exp]        = speeds_on.flatten()
                speeds_on_norm[scen][exp]   = speeds_on_all[scen][exp] / np.ma.median(speeds_ref[scen][exp])
                speeds_off_all[scen][exp]       = speeds_off.flatten()
                speeds_off_norm[scen][exp]  = speeds_off_all[scen][exp] / np.ma.median(speeds_ref[scen][exp])
                
                ang_vel_ref[scen][exp]  = ang_vel[0:20].flatten()
                ang_vel_norm[scen][exp] = ang_vel[20:].flatten()
                ang_vel_norm[scen][exp] = ang_vel_norm[scen][exp] / np.ma.median(ang_vel_ref[scen][exp])
                
                ang_vel_on_all[scen][exp]       = ang_vel_on.flatten()
                ang_vel_on_norm[scen][exp]  = ang_vel_on_all[scen][exp] / np.ma.median(ang_vel_ref[scen][exp])
                ang_vel_off_all[scen][exp]      = ang_vel_off.flatten()
                ang_vel_off_norm[scen][exp] = ang_vel_off_all[scen][exp] / np.ma.median(ang_vel_ref[scen][exp])
                
                # averages over the agents
                speeds_mean_ref[scen][exp] = np.ma.mean(speeds[0:20], axis=1)
                speeds_mean_norm[scen][exp] = np.ma.mean(speeds[20:], axis=1)
                speeds_mean_norm[scen][exp] = speeds_mean_norm[scen][exp] / np.ma.median(speeds_mean_ref[scen][exp])
                
                speeds_mean_on_norm[scen][exp] = np.ma.mean(speeds_on, axis=1)
                speeds_mean_on_norm[scen][exp] = speeds_mean_on_norm[scen][exp] / np.ma.median(speeds_mean_ref[scen][exp])
                speeds_mean_off_norm[scen][exp] = np.ma.mean(speeds_off, axis=1)
                speeds_mean_off_norm[scen][exp] = speeds_mean_off_norm[scen][exp] / np.ma.median(speeds_mean_ref[scen][exp])
                speeds_mean_on_norm2[scen][exp] = np.ma.mean(speeds_on, axis=1) / np.median(np.ma.mean(speeds_off, axis=1))
                speeds_mean_off_norm2[scen][exp] = np.ma.mean(speeds_off, axis=1) / np.median(np.ma.mean(speeds_off, axis=1))
                
                ang_vel_mean_ref[scen][exp] = np.ma.mean(ang_vel[0:20], axis=1)
                ang_vel_mean_norm[scen][exp] = np.ma.mean(ang_vel[20:], axis=1)
                ang_vel_mean_norm[scen][exp] = ang_vel_mean_norm[scen][exp] / np.ma.median(ang_vel_mean_ref[scen][exp])
                
                ang_vel_mean_on_norm[scen][exp] = np.ma.mean(ang_vel_on, axis=1)
                ang_vel_mean_on_norm[scen][exp] = ang_vel_mean_on_norm[scen][exp] / np.ma.median(ang_vel_mean_ref[scen][exp])
                ang_vel_mean_off_norm[scen][exp] = np.ma.mean(ang_vel_off, axis=1)
                ang_vel_mean_off_norm[scen][exp] = ang_vel_mean_off_norm[scen][exp] / np.ma.median(ang_vel_mean_ref[scen][exp])
                ang_vel_mean_on_norm2[scen][exp] = np.ma.mean(ang_vel_on, axis=1) / np.median(np.ma.mean(ang_vel_off, axis=1))
                ang_vel_mean_off_norm2[scen][exp] = np.ma.mean(ang_vel_off, axis=1) / np.median(np.ma.mean(ang_vel_off, axis=1))
                
    #aggregate data from the same scenario
    speeds_all_norm_aggregate =         [np.ma.concatenate(d) for d in speeds_norm]
    speeds_all_on_aggregate =           [np.ma.concatenate(d) for d in speeds_on_all]
    speeds_all_on_norm_aggregate =      [np.ma.concatenate(d) for d in speeds_on_norm]
    speeds_all_off_aggregate =          [np.ma.concatenate(d) for d in speeds_off_all]
    speeds_all_off_norm_aggregate =     [np.ma.concatenate(d) for d in speeds_off_norm]
    ang_vel_all_norm_aggregate =        [np.ma.concatenate(d) for d in ang_vel_norm]
    ang_vel_all_on_aggregate =          [np.ma.concatenate(d) for d in ang_vel_on_all]
    ang_vel_all_on_norm_aggregate =     [np.ma.concatenate(d) for d in ang_vel_on_norm]
    ang_vel_all_off_aggregate =         [np.ma.concatenate(d) for d in ang_vel_off_all]
    ang_vel_all_off_norm_aggregate =    [np.ma.concatenate(d) for d in ang_vel_off_norm]
    
    speeds_mean_norm_aggregate =        [np.ma.concatenate(d) for d in speeds_mean_norm]
    ang_vel_mean_norm_aggregate =       [np.ma.concatenate(d) for d in ang_vel_mean_norm]
    speeds_mean_on_norm_aggregate =     [np.ma.concatenate(d) for d in speeds_mean_on_norm]
    speeds_mean_off_norm_aggregate =    [np.ma.concatenate(d) for d in speeds_mean_off_norm]
    ang_vel_mean_on_norm_aggregate =    [np.ma.concatenate(d) for d in ang_vel_mean_on_norm]
    ang_vel_mean_off_norm_aggregate =   [np.ma.concatenate(d) for d in ang_vel_mean_off_norm]
    speeds_mean_on_norm2_aggregate =    [np.ma.concatenate(d) for d in speeds_mean_on_norm2]
    speeds_mean_off_norm2_aggregate =   [np.ma.concatenate(d) for d in speeds_mean_off_norm2]
    ang_vel_mean_on_norm2_aggregate =   [np.ma.concatenate(d) for d in ang_vel_mean_on_norm2]
    ang_vel_mean_off_norm2_aggregate =  [np.ma.concatenate(d) for d in ang_vel_mean_off_norm2]
    
    print('scen\tspeed OFF\tspeed ON\t\tang.v. OFF\tang.v.ON')
    for scen in range(number_of_scenarios):
        if labels: 
            print(labels[scen],end='\t')
        else: 
            print(f'{scen}',end='\t\t')
        mean_off = np.mean(speeds_all_off_aggregate[scen])
        std_off  = np.std(speeds_all_off_aggregate[scen])
        mean_on = np.mean(speeds_all_on_aggregate[scen])
        std_on  = np.std(speeds_all_on_aggregate[scen])
        print(f'{mean_off:.2f}±{std_off:.2f}',end='\t')
        print(f'{mean_on:.2f}±{std_on:.2f}',end='\t')
        print(f'{((mean_on-mean_off)/mean_off*100):+.0f}%',end='\t')
        mean_off = np.mean(ang_vel_all_off_aggregate[scen])
        std_off  = np.std(ang_vel_all_off_aggregate[scen])
        mean_on = np.mean(ang_vel_all_on_aggregate[scen])
        std_on  = np.std(ang_vel_all_on_aggregate[scen])
        print(f'{mean_off:.2f}±{std_off:.2f}',end='\t')
        print(f'{mean_on:.2f}±{std_on:.2f}',end='\t')
        print(f'{((mean_on-mean_off)/mean_off*100):+.0f}%',end='\t')
        print('')
    
    # PLOTS
    plots_dir = '/Volumes/DOMEPEN/Experiments/comparisons'
    with open(os.path.join(plots_dir, 'scenario_info.txt'), 'w') as file:
        file.write(f'Experiments Comparison {labels}\nExperiments: {experiment_names}\nTracking: {tracking_folders}')
    
    # comparison speed and ang vel of all values
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    #my_boxplot([speeds_all_norm_aggregate, speeds_all_on_norm_aggregate, speeds_all_off_norm_aggregate], whis = 3)
    #plot_with_shade(xvalues, speeds_all_norm_aggregate, quantile=[0.25, 0.75])
    p_on = plot_with_shade(xvalues, speeds_all_on_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, speeds_all_off_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.ylabel('Speed/Median speed first 10s')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.title('All values')
    plt.xticks(xvalues,labels)
    plt.subplot(2, 1, 2)
    #plot_with_shade(xvalues, ang_vel_all_norm_aggregate, quantile=[0.25, 0.75])
    p_on = plot_with_shade(xvalues, ang_vel_all_on_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, ang_vel_all_off_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.ylabel('Ang Vel/Median ang vel first 10s')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.xticks(xvalues,labels)
    plt.savefig(os.path.join(plots_dir, 'scenario_all_normalized.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, mean
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plot_with_shade(xvalues, speeds_mean_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.xticks(xvalues,labels)
    plt.ylabel('Speed/Median speed first 10s')
    plt.title('Averages over agents')
    plt.subplot(2, 1, 2)
    plot_with_shade(xvalues, ang_vel_mean_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.xticks(xvalues,labels)
    plt.ylabel('Ang Vel/Median ang vel first 10s')
    plt.savefig(os.path.join(plots_dir, 'scenario_mean_normalized_mean.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, on vs off
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    p_on = plot_with_shade(xvalues, speeds_mean_on_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, speeds_mean_off_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.ylabel('Speed/Median speed first 10s')
    plt.title('Averages over agents')
    plt.xticks(xvalues,labels)
    plt.subplot(2, 1, 2)
    p_on = plot_with_shade(xvalues, ang_vel_mean_on_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, ang_vel_mean_off_norm_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.ylabel('Ang Vel/Median ang vel first 10s')
    plt.xticks(xvalues,labels)
    plt.savefig(os.path.join(plots_dir, 'scenario_mean_normalized_on_off.pdf'), bbox_inches = 'tight')
    plt.show()
    
    # comparison speed and ang vel of averages over agents, on vs off
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    p_on = plot_with_shade(xvalues, speeds_mean_on_norm2_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, speeds_mean_off_norm2_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.ylabel('Speed/Median OFF speed')
    plt.title('Averages over agents')
    plt.xticks(xvalues,labels)
    plt.subplot(2, 1, 2)
    p_on = plot_with_shade(xvalues, ang_vel_mean_on_norm2_aggregate, quantile=[0.25, 0.75], marker='o')
    p_off = plot_with_shade(xvalues, ang_vel_mean_off_norm2_aggregate, quantile=[0.25, 0.75], marker='o')
    plt.legend([p_on,p_off],['Light ON','Light OFF'])
    plt.ylabel('Ang Vel/Median OFF ang vel')
    plt.xticks(xvalues,labels)
    plt.savefig(os.path.join(plots_dir, 'scenario_mean_normalized2_on_off.pdf'), bbox_inches = 'tight')
    plt.show()
    

# MAIN -------------------------------------------------------------------------------------
# experiments_directory = '/Users/andrea/Library/CloudStorage/OneDrive-UniversitàdiNapoliFedericoII/Andrea_Giusti/Projects/DOME/Experiments'
# experiments_directory = '\\\\tsclient\DOMEPEN\Experiments'
experiments_directory = '/Volumes/DOMEPEN/Experiments'
# experiments_directory = 'D:\AndreaG_DATA\Experiments'

experiment_name = "2023_06_15_Euglena_1"

off_experiments=['2023_06_15_Euglena_1','2023_06_26_Euglena_13', '2023_07_10_Euglena_6']

on255_experiments=['2023_06_15_Euglena_4','2023_06_26_Euglena_19', '2023_07_10_Euglena_12']
on150_experiments=['2023_06_15_Euglena_3','2023_06_26_Euglena_17', '2023_07_10_Euglena_10']
on75_experiments=['2023_06_15_Euglena_2','2023_06_26_Euglena_15', '2023_07_10_Euglena_8']

switch10_experiments=['2023_06_15_Euglena_7','2023_06_26_Euglena_23', '2023_07_10_Euglena_15']
switch5_experiments=['2023_06_15_Euglena_8','2023_06_26_Euglena_25', '2023_07_10_Euglena_18']

#tracking_folder ='tracking_2023_10_09'
tracking_folder ='last'

# parameters
min_traj_length = 5     # minimum length of the trajectories [s]
variance_thresh = 2.5   # variance threshold for outliers detection


print('Now use one of the following commands:'
       '\n\tanalyse_experiment(experiment_name)'
       '\n\texperiments_comparison(experiment_names)'
       '\n\tscenarios_comparison(experiment_names)')

# scenarios_comparison([on255_experiments, switch10_experiments, switch5_experiments],['60s','10s','5s'], [ 100/60,100/10,100/5])
# scenarios_comparison([on75_experiments,on150_experiments,on255_experiments],['30%','60%','100%'], [30,60,100])





