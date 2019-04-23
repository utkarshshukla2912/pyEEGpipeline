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

		with open('support/config.json') as f:
			config = json.load(f)

		self.preProcessedData = config['preProcessedData']
		self.samplingFrequency = config['samplingFrequency']
		self.featureExtracted = config['featureExtracted']
		self.dataFeatures = config['dataFeatures']
		self.objectPath = config['objectPath']
		self.windowSize = config['windowSize']
		self.channels = config['channels']
		self.bands = config['bands']
		self.channel_data = []



		paths = [self.objectPath,self.objectPath+self.preProcessedData,
				self.objectPath+self.dataFeatures]
		for path in paths:
			if not os.path.exists(path):
				os.mkdir(path)

	def BandPower(self):
		resp = pyeeg.bin_power(self.channel_data,self.bands,self.samplingFrequency)
		resp = np.array(resp).flatten()
		label = 'bandPower_'
		labels = label+pd.Series(range(len(resp)),dtype=str)
		return [resp,labels.values]

	def SpectralEntropy(self):
		b = pyeeg.bin_power(self.channel_data,self.bands,self.samplingFrequency)
		resp = pyeeg.spectral_entropy(self.channel_data,self.bands,self.samplingFrequency,Power_Ratio=b)
		resp = [0 if math.isnan(x) else x for x in resp]
		label = 'SpectralEntropy_'
		labels = label+pd.Series(range(len(resp)),dtype=str)
		return [np.array(resp),labels.values]

	def CorrelationDimension(self):
		resp = nolds.corr_dim(self.channel_data,1)
		return [np.array([resp]),['CorrelationDimension']]

	def DFA(self):
		resp = pyeeg.dfa(self.channel_data)
		return [np.array([resp]),['DFA']]

	def FirstDiff(self):
		resp = pyeeg.first_order_diff(self.channel_data)
		label = 'FirstDiff_'
		labels = label+pd.Series(range(len(resp)),dtype=str)
		return [resp,labels.values]

	def Hjorth(self):
		resp = pyeeg.hjorth(self.channel_data)
		label = 'Hjorth_'
		labels = label+pd.Series(range(len(resp)),dtype=str)
		return [np.array(resp),labels.values]

	def Hurst(self):
		resp = pyeeg.hurst(self.channel_data)
		return [np.array([resp]),['hurst']]

	def Mean(self):
		resp = np.mean(self.channel_data)
		return [np.array([resp]),['mean']]

	def PFD(self):
		resp = pyeeg.pfd(self.channel_data)
		return [np.array([resp]),['PFD']]

	def Power(self):
		F = np.fft.fft(self.channel_data)
		P = F * np.conjugate(F)
		resp = sum(P)
		return [np.array([abs(resp)]),['Power']]

	def Std(self):
		resp = np.std(self.channel_data)
		return [np.array([resp]),['STD']]

	def DWT(self):
		resp = pywt.dwt(self.channel_data, 'db4')
		resp = np.array(resp).flatten()
		label = 'DWT_'
		labels = label+pd.Series(range(len(resp)),dtype=str)
		return [resp,labels.values]

	def runPipeline(self):
		functions = np.array([self.BandPower(),self.SpectralEntropy(),self.CorrelationDimension(),
		self.DFA(),self.Hjorth(),self.Hurst(),self.Mean(),self.PFD(),self.Power(),
		self.Std(),self.DWT()])
		features = []
		headers = []
		for feature in functions[:,0]:
			for val in feature:
				features.append(val)
		for header in functions[:,1]:
			for val in header:
				headers.append(val)
		return [features,headers]

	def getFeatures(self):
		files = glob.glob(self.objectPath+self.preProcessedData+'*.npy')
		split_length = None

		if self.windowSize != "None":
			split_length = self.windowSize * self.samplingFrequency

		split_based = open(self.objectPath+self.dataFeatures+self.featureExtracted, 'w', newline='')
		writer = csv.writer(split_based, delimiter=',')
		header_writen = False
		for file in tqdm(files):
			file_split = file.split('_')
			recording_class = file_split[2]
			recording = np.load(file)
			i = 0
			for channel in tqdm(recording):
				if self.windowSize == "None":
					split_length = len(channel)
				limit = int(len(channel)/split_length)*split_length
				channel = channel[0:limit]
				splits = np.split(channel,limit//split_length)
				j = 1
				for split in tqdm(splits):
					self.channel_data = split
					data_ = self.runPipeline()
					temp = [file_split[0],recording_class,self.channels[i],j]
					features = list(data_[0])
					if not header_writen:
						writer.writerow(  ['filename','experiment_identifier','channel_name','split_number'] + list(data_[1]) )
						header_writen = True
					writer.writerow(temp+features)
					#break
					j += 1
					#break
				i += 1
				#break



#done



#featureExtractor().getFeatures()

#done
