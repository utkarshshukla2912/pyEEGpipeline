from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as sps
from tqdm import tqdm
import numpy as np
import mne.io
import glob
import json
import mne
import os


class loadData:


    def __init__(self):

        with open('config.json') as f:
            config = json.load(f)

        self.preprocessed_dataV1 = config['preprocessed_dataV1']
        self.samplingFrequency = config['samplingFrequency']
        self.objectPath = config['objectPath']
        self.channels = config['channels']
        self.dataPath = config['dataPath']
        self.highcut = config['highcut']
        self.lowcut = config['lowcut']
        self.label = []
        self.data = []

        paths = [self.dataPath, self.objectPath,self.objectPath+self.preprocessed_dataV1]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)


    def butter_bandpass(self,order=5):
        nyq = 0.5 * self.samplingFrequency
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self,data,order=5):
        b, a = self.butter_bandpass(order=order)
        y = lfilter(b, a, data)
        return y


    def readData(self):
        folders = glob.glob(self.dataPath+'*/')
        for folder in tqdm(folders):
            files = glob.glob(folder+'/*.edf')
            for file in tqdm(files):
                id = file.split('_')[1]
                recording_class = file.split('_')[2]
                if recording_class == 'Emotion' or recording_class == 'Baseline':
                    recording_class += '_'+file.split('_')[3]

                raw_data = mne.io.read_raw_edf(file,preload=True,verbose=50)
                raw_data.set_montage('standard_1020',verbose=50)

                channel_specific = raw_data.pick_channels(self.channels)
                band_filter_data = self.butter_bandpass_filter(channel_specific.get_data())
                np.save(self.objectPath+self.preprocessed_dataV1+"{}_{}.npy".format(id,recording_class),
                        band_filter_data)
