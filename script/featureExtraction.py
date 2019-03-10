from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import logging
import pyeeg
import nolds
import math
import pywt
import json
import glob
import csv
import os
warnings.filterwarnings('ignore')




class featureExtractor:

    def __init__(self):

        with open('config.json') as f:
            config = json.load(f)

        #self.preprocessed_dataV2 = config['preprocessed_dataV2']
        self.preprocessed_dataV1 = config['preprocessed_dataV1']
        self.samplingFrequency = config['samplingFrequency']
        self.dataFeatures = config['dataFeatures']
        self.objectPath = config['objectPath']
        self.windowSize = config['windowSize']
        self.channels = config['channels']
        self.bands = config['bands']
        self.channel_data = []

        paths = [self.objectPath,self.objectPath+self.preprocessed_dataV1,
                self.objectPath+self.dataFeatures]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

    def BandPower(self):
        resp = pyeeg.bin_power(self.channel_data,self.bands,self.samplingFrequency)
        return np.array(resp).flatten()

    def SpectralEntropy(self):
        b = pyeeg.bin_power(self.channel_data,self.bands,self.samplingFrequency)
        resp = pyeeg.spectral_entropy(self.channel_data,self.bands,self.samplingFrequency,Power_Ratio=b)
        resp = [0 if math.isnan(x) else x for x in resp]
        return np.array(resp)

    def CorrelationDimension(self):
        resp = nolds.corr_dim(self.channel_data,1)
        return np.array([resp])

    def DFA(self):
    	resp = pyeeg.dfa(self.channel_data)
    	return np.array([resp])

    def FirstDiff(self):
    	resp = pyeeg.first_order_diff(self.channel_data)
    	return resp

    def Hjorth(self):
    	resp = pyeeg.hjorth(self.channel_data)
    	return np.array(resp)

    def Hurst(self):
    	resp = pyeeg.hurst(self.channel_data)
    	return np.array([resp])

    def Mean(self):
    	resp = np.mean(self.channel_data)
    	return np.array([resp])

    def PFD(self):
    	resp = pyeeg.pfd(self.channel_data)
    	return np.array([resp])

    def Power(self):
    	F = np.fft.fft(self.channel_data)
    	P = F * np.conjugate(F)
    	resp = sum(P)
    	return np.array([abs(resp)])

    def Std(self):
    	resp = np.std(self.channel_data)
    	return np.array([resp])

    def DWT(self):
    	resp = pywt.dwt(self.channel_data, 'db4')
    	return np.array(resp).flatten()

    def runPipeline(self):
        functions =[self.BandPower(),self.SpectralEntropy(),self.CorrelationDimension(),
        self.DFA(),self.Hjorth(),self.Hurst(),self.Mean(),self.PFD(),self.Power(),
        self.Std(),self.DWT()]
        features = []
        for feature in functions:
            for val in feature:
                features.append(val)
        return features
        #self.FirstDiff()

    def fetchData(self):
        files = glob.glob(self.objectPath+self.preprocessed_dataV1+'*.npy')
        split_length = self.windowSize * self.samplingFrequency
        split_based = open(self.objectPath+self.dataFeatures+'channel_split_features.csv', 'w', newline='')
        channel_based = open(self.objectPath+self.dataFeatures+'channel_features.csv', 'w', newline='')
        writer1 = csv.writer(split_based, delimiter=',')
        writer2 = csv.writer(channel_based, delimiter=',')
        for file in tqdm(files):
            file_values = file[file.rindex('/')+1:].split('_')
            username = file_values[0]
            username = username
            recording_class = '_'.join(file_values[1:]).replace('.npy','')
            recording = np.load(file)
            i = 0
            for channel in tqdm(recording):
                #self.channel_data = channel
                #feature = self.runPipeline()
                #writer2.writerow([username,recording_class,self.channels[i],feature])
                splits = np.array_split(channel,split_length)
                j = 1
                for split in tqdm(splits):
                    self.channel_data = split
                    feature = self.runPipeline()
                    writer1.writerow([username,recording_class,self.channels[i],j,feature])
                    j += 1
                i += 1


featureExtractor().fetchData()

#done
