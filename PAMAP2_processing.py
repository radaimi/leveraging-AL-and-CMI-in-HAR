#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:31:56 2018

@author: Rebecca Adaimi

PAMAP2 Dataset Preprocessing
"""

import numpy as np
import pandas as pd
import os
import math as m
import matplotlib.pyplot as plt 
from scipy import stats
import scipy.fftpack 
import copy
import scipy as sp
import scipy.signal
from collections import Counter

sampling_freq = 100  # 100Hz


def plot_fft(data, fs, isTrue):
    # Number of samplepoints
    N = len(data)
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N*T, N)
    y = data
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    if isTrue:
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), 'k')
    else:
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth=0.3)


def plot_data(p, s, side, x, y, label = None):
    plt.figure()
    plt.plot(x,y, linewidth=0.3)
    if label is not None:
        plt.plot(x, y*label)
        
    plt.title(str(p) + '_' + s + '_acc_' + side)    
    plt.show()
        
def gen_features_multimodal(hand, ankle, chest):
    feature_mat = np.empty((0,48))
    
    # hand+ankle
    data_x = hand[:,0] + ankle[:,0]
    data_y = hand[:,1] + ankle[:,1]
    data_z = hand[:,2] + ankle[:,2]
    feature1 = np.mean(data_x)
    feature2 = np.mean(data_y)
    feature3 = np.mean(data_z)
    feature4 = np.absolute(np.trapz(data_x))
    feature5 = np.absolute(np.trapz(data_y))
    feature6 = np.absolute(np.trapz(data_z))
    feature7 = np.std(data_x)
    feature8 = np.std(data_y)
    feature9 = np.std(data_z)
    feature10 = np.sum(data_x*data_x)
    feature11 = np.sum(data_y*data_y)
    feature12 = np.sum(data_z*data_z)
    
    # hand+chest
    data_x = hand[:,0] + chest[:,0]
    data_y = hand[:,1] + chest[:,1]
    data_z = hand[:,2] + chest[:,2]
    feature13 = np.mean(data_x)
    feature14 = np.mean(data_y)
    feature15 = np.mean(data_z)
    feature16 = np.absolute(np.trapz(data_x))
    feature17 = np.absolute(np.trapz(data_y))
    feature18 = np.absolute(np.trapz(data_z))
    feature19 = np.std(data_x)
    feature20 = np.std(data_y)
    feature21 = np.std(data_z)
    feature22 = np.sum(data_x*data_x)
    feature23 = np.sum(data_y*data_y)
    feature24 = np.sum(data_z*data_z)
    
    #ankle+chest
    data_x = ankle[:,0] + chest[:,0]
    data_y = ankle[:,1] + chest[:,1]
    data_z = ankle[:,2] + chest[:,2]
    feature25 = np.mean(data_x)
    feature26 = np.mean(data_y)
    feature27 = np.mean(data_z)
    feature28 = np.absolute(np.trapz(data_x))
    feature29 = np.absolute(np.trapz(data_y))
    feature30 = np.absolute(np.trapz(data_z))
    feature31 = np.std(data_x)
    feature32 = np.std(data_y)
    feature33 = np.std(data_z)
    feature34 = np.sum(data_x*data_x)
    feature35 = np.sum(data_y*data_y)
    feature36 = np.sum(data_z*data_z) 
    
    #hand+chest+ankle
    data_x = ankle[:,0] + chest[:,0] + hand[:,0]
    data_y = ankle[:,1] + chest[:,1] + hand[:,1]
    data_z = ankle[:,2] + chest[:,2] + hand[:,2]
    feature37 = np.mean(data_x)
    feature38 = np.mean(data_y)
    feature39 = np.mean(data_z)
    feature40 = np.absolute(np.trapz(data_x))
    feature41 = np.absolute(np.trapz(data_y))
    feature42 = np.absolute(np.trapz(data_z))
    feature43 = np.std(data_x)
    feature44 = np.std(data_y)
    feature45 = np.std(data_z)
    feature46 = np.sum(data_x*data_x)
    feature47 = np.sum(data_y*data_y)
    feature48 = np.sum(data_z*data_z)    
    
    feature_mat = np.array([feature1, feature2, feature3, feature4, feature5, 
                            feature6, feature7, feature8, feature9, feature10, 
                            feature11, feature12, feature13, feature14, feature15, 
                            feature16, feature17, feature18, feature19, feature20, 
                            feature21, feature22, feature23, feature24, feature25, 
                            feature26, feature27, feature28, feature29, feature30, 
                            feature31, feature32, feature33, feature34, feature35, 
                            feature36, feature37, feature38, feature39, feature40,
                            feature41, feature42, feature43, feature44, feature45,
                            feature46, feature47, feature48])
    #print(feature_mat)
    return feature_mat    
    

def gen_features_HR(data, data_n):
    
    feature_mat = np.empty((0,4))
    feature1 = np.mean(data)
    feature2 = np.mean(np.gradient(data))
       
    feature3 = np.mean(data_n)
    feature4 = np.mean(np.gradient(data_n))   
    
    feature_mat = np.array([feature1, feature2, feature3, feature4])
    
    return feature_mat

def gen_features(data_x, data_y, data_z):
    feature_mat = np.empty((0,33))
    
   # (mean, median, standard deviation, peak acceleration and energy)
    feature1 = np.mean(data_x)
    feature2 = np.mean(data_y)
    feature3 = np.mean(data_z)
    feature4 = np.median(data_x)
    feature5 = np.median(data_y)
    feature6 = np.median(data_z)
    feature7 = np.std(data_x)
    feature8 = np.std(data_y)
    feature9 = np.std(data_z)
    feature10 = np.max(data_x)
    feature11 = np.max(data_y)
    feature12 = np.max(data_z)
    feature13 = np.sum(data_x*data_x)
    feature14 = np.sum(data_y*data_y)
    feature15 = np.sum(data_z*data_z)
    
    # absolute integral
    feature16 = np.absolute(np.trapz(data_x))
    feature17 = np.absolute(np.trapz(data_y))
    feature18 = np.absolute(np.trapz(data_z))
    
    # Correlation between each pair of axes, 
    feature19 = np.corrcoef(data_x, data_y)[0,1]
    feature20 = np.corrcoef(data_y, data_z)[0,1]
    feature21 = np.corrcoef(data_x, data_z)[0,1]
    
    # Power ratio of the frequency bands 0–2.75 Hz and 0–5 Hz
    fx, Pxx = scipy.signal.periodogram(data_x, sampling_freq)
    total_power = sp.trapz(Pxx, fx)
    
    ind_min = sp.argmax(fx > 0) - 1
    ind_max = sp.argmax(fx > 2.75) - 1
    bandpower = sp.trapz(Pxx[ind_min: ind_max], fx[ind_min: ind_max])
    feature22 = float(bandpower)/float(total_power)

    ind_min = sp.argmax(fx > 0) - 1
    ind_max = sp.argmax(fx > 5) - 1
    bandpower = sp.trapz(Pxx[ind_min: ind_max], fx[ind_min: ind_max])
    feature23 = float(bandpower)/float(total_power)
    
    fy, Pyy = scipy.signal.periodogram(data_y, sampling_freq)
    total_power = sp.trapz(Pyy, fy)
    
    ind_min = sp.argmax(fy > 0) - 1
    ind_max = sp.argmax(fy > 2.75) - 1
    bandpower = sp.trapz(Pyy[ind_min: ind_max], fy[ind_min: ind_max])
    feature24 = float(bandpower)/float(total_power)

    ind_min = sp.argmax(fy > 0) - 1
    ind_max = sp.argmax(fy > 5) - 1
    bandpower = sp.trapz(Pyy[ind_min: ind_max], fy[ind_min: ind_max])
    feature25 = float(bandpower)/float(total_power)

    fz, Pzz = scipy.signal.periodogram(data_z, sampling_freq)
    total_power = sp.trapz(Pzz, fz)
    
    ind_min = sp.argmax(fz > 0) - 1
    ind_max = sp.argmax(fz > 2.75) - 1
    bandpower = sp.trapz(Pzz[ind_min: ind_max], fz[ind_min: ind_max])
    feature26 = float(bandpower)/float(total_power)

    ind_min = sp.argmax(fz > 0) - 1
    ind_max = sp.argmax(fz > 5) - 1
    bandpower = sp.trapz(Pzz[ind_min: ind_max], fz[ind_min: ind_max])
    feature27 = float(bandpower)/float(total_power)
    # peak frequency of the power spectral density
    feature28 = fx[np.argmax(Pxx)]
    feature29 = fy[np.argmax(Pyy)]
    feature30 = fz[np.argmax(Pzz)]
    
    # Spectral entropy of the normalized PSD
    Pxx_n = Pxx/np.sum(Pxx)
    Pyy_n = Pyy/np.sum(Pyy)
    Pzz_n = Pzz/np.sum(Pzz)
    
    logPxx = np.log2(Pxx_n + 1e-12)
    logPyy = np.log2(Pyy_n + 1e-12)
    logPzz = np.log2(Pzz_n + 1e-12)

    feature31 = (-1)*np.sum(Pxx_n*logPxx)
    feature32 = (-1)*np.sum(Pyy_n*logPyy)
    feature33 = (-1)*np.sum(Pzz_n*logPzz)
    
    feature_mat = np.array([feature1, feature2, feature3, feature4, feature5, 
                            feature6, feature7, feature8, feature9, feature10, 
                            feature11, feature12, feature13, feature14, feature15, 
                            feature16, feature17, feature18, feature19, feature20, 
                            feature21, feature22, feature23, feature24, feature25, 
                            feature26, feature27, feature28, feature29, feature30, 
                            feature31, feature32, feature33])
    #print(feature_mat)
    return feature_mat    
        
if __name__ == "__main__":   

    path = './datasets/PAMAP2_Dataset/Protocol/' 
    participants = os.listdir(path)

    mat = np.empty((0,86))  
    window_size = int(5.12*sampling_freq)
    overlap = 1*sampling_freq # 1s overlap
    feature_path = './datasets/PAMAP2_Dataset/Features'
    for p in sorted(participants):
        
        mat_participant = np.empty((0,86))
    
        print (str(p))
        full_path = os.sep.join((path,p))
        data = pd.DataFrame(np.genfromtxt(full_path, skip_header=1, skip_footer=1, names=True, dtype=None, delimiter=' '))            
        
        data.iloc[:,2] = data.iloc[:,2].interpolate()
        
        transient = np.where(data.iloc[:,1] == 0)[0]
        data = np.delete(np.array(data), transient,axis=0)
        
        data = data[~np.isnan(data).any(axis=1)]

        label = data[:,1]

        # Heart Rate Data
        HR_data = data[:,2]
        
        # normalized HR Data
        HR_data_n = HR_data - np.mean(HR_data[np.argmin(HR_data):np.argmax(HR_data)])
        HR_data_n = HR_data/np.std(HR_data[np.argmin(HR_data):np.argmax(HR_data)])

        
        # Hand data
        acc_hand_data = data[:,4:7]
        gyro_hand_data = data[:,10:13]
        mag_hand_data = data[:,13:16]
        
        # Chest Data
        acc_chest_data = data[:,21:24]
        gyro_chest_data = data[:,27:30]
        mag_chest_data = data[:,30:33]
        
        # Ankle Data
        acc_ankle_data = data[:,37:40]
        gyro_ankle_data = data[:,43:46]
        mag_ankle_data = data[:,46:49]
        
        #print ("Loaded Data")

        # sliding window
        start_index = 0
        end_index = start_index + window_size
        
        while end_index <= len(data) and start_index < len(data):
            mat_feat = []
            acc_hand_segment = acc_hand_data[start_index:end_index,:]
            # gyro_hand_segment = gyro_hand_data[start_index:end_index,:]
            # mag_hand_segment = mag_hand_data[start_index:end_index,:]
            
            acc_chest_segment = acc_chest_data[start_index:end_index,:]
            # gyro_chest_segment = gyro_chest_data[start_index:end_index,:]
            # mag_chest_segment = mag_chest_data[start_index:end_index,:]
            
            acc_ankle_segment = acc_ankle_data[start_index:end_index,:]
            # gyro_ankle_segment = gyro_ankle_data[start_index:end_index,:]
            # mag_ankle_segment = mag_ankle_data[start_index:end_index,:]
            
            HR_segment = HR_data[start_index:end_index]
            HR_segment_n = HR_data_n[start_index:end_index]
            
            label_segment = label[start_index:end_index]

            mat_feat = np.hstack((mat_feat, gen_features(acc_hand_segment[:,0], acc_hand_segment[:,1], acc_hand_segment[:,2])))
            mat_feat = np.hstack((mat_feat, gen_features_multimodal(acc_hand_segment, acc_chest_segment, acc_ankle_segment)))
            mat_feat = np.hstack((mat_feat, gen_features_HR(HR_segment, HR_segment_n)))
            
            Counter(label_segment)
            mat_feat = np.hstack((mat_feat, Counter(label_segment).most_common(1)[0][0]))
            
            #print(np.shape(mat),np.shape(np.reshape(mat_feat,(1,-1))))
            mat = np.vstack((mat , np.reshape(mat_feat,(1,-1))))
            mat_participant = np.vstack((mat_participant, np.reshape(mat_feat,(1,-1))))

            
            start_index = end_index - overlap + 1
            end_index = start_index + window_size

        
        end_index = len(data)
        mat_feat = []
        acc_hand_segment = acc_hand_data[start_index:end_index,:]
        # gyro_hand_segment = gyro_hand_data[start_index:end_index,:]
        # mag_hand_segment = mag_hand_data[start_index:end_index,:]
        
        acc_chest_segment = acc_chest_data[start_index:end_index,:]
        # gyro_chest_segment = gyro_chest_data[start_index:end_index,:]
        # mag_chest_segment = mag_chest_data[start_index:end_index,:]
        
        acc_ankle_segment = acc_ankle_data[start_index:end_index,:]
        # gyro_ankle_segment = gyro_ankle_data[start_index:end_index,:]
        # mag_ankle_segment = mag_ankle_data[start_index:end_index,:]
        
        HR_segment = HR_data[start_index:end_index]
        HR_segment_n = HR_data_n[start_index:end_index]
        
        label_segment = label[start_index:end_index]

        mat_feat = np.hstack((mat_feat, gen_features(acc_hand_segment[:,0], acc_hand_segment[:,1], acc_hand_segment[:,2])))
        mat_feat = np.hstack((mat_feat, gen_features_multimodal(acc_hand_segment, acc_chest_segment, acc_ankle_segment)))
        mat_feat = np.hstack((mat_feat, gen_features_HR(HR_segment, HR_segment_n)))
        
        Counter(label_segment)
        mat_feat = np.hstack((mat_feat, Counter(label_segment).most_common(1)[0][0]))
        
        #print(np.shape(mat),np.shape(np.reshape(mat_feat,(1,-1))))
        mat = np.vstack((mat , np.reshape(mat_feat,(1,-1))))
        mat_participant = np.vstack((mat_participant, np.reshape(mat_feat,(1,-1))))            
        

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        pd.DataFrame(mat_participant).to_csv(feature_path + "/PAMAP2_" + str(p[:-3]) + "features.csv", header=False,index=False)
                
                
                
                    

                
                
                
                
        
        
    
    

