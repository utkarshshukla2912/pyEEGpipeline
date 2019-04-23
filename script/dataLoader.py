from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter
from pyprep.noisy import Noisydata
import matplotlib.pyplot as plt
import scipy.signal as sps
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import mne.io
import glob
import json
import mne
import os
warnings.filterwarnings('ignore')

class dataLoader:


    def __init__(self):

        with open('support/config.json') as f:
            config = json.load(f)

        self.samplingFrequency = config['samplingFrequency']
        self.preProcessedData = config['preProcessedData']
        self.objectPath = config['objectPath']
        self.channels = config['channels']
        self.dataPath = config['dataPath']
        self.fileType = config['fileType']
        self.highcut = config['highcut']
        self.lowcut = config['lowcut']
        self.readingFunction = None
        self.label = []
        self.data = []

        paths = [self.dataPath, self.objectPath,
                 self.objectPath+self.preProcessedData]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

        if self.fileType == 'edf':
            self.readingFunction = self.readEdf
        elif self.fileType == 'mat':
            self.readingFunction = self.readMat
        else:
            self.readingFunction = self.readCsv


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

    def get_bad_channels(self,rawData,sfreq = 128,n_chans = 14):
        nd = Noisydata(rawData)
        nd.find_all_bads(ransac=False)
        bads = nd.get_bads(verbose=False)
        return bads

    def readCsv(self,file):
        data = pd.read_csv(file,header=None)
        data  = data[:4]
        info = mne.create_info(ch_names=self.channels,
                               sfreq=self.samplingFrequency,
                               ch_types=['eeg']*len(self.channels))
        raw = mne.io.RawArray(np.array(data), info, verbose = 50)
        return raw

    def readEdf(self,file):
        raw = mne.io.read_raw_edf(file,verbose=50)
        return raw

    def readMat(self,file):
        return 0

    def loadData(self):
        folders = glob.glob(self.dataPath+'*/')
        for folder in tqdm(folders):
            files = glob.glob(folder+'/*.{}'.format(self.fileType))
            if folder[-1] != '/':
                label = folder[folder.rindex('/')+1:]
            else:
                label = folder[folder[:-1].rindex('/')+1:-1]
            counter = 1
            for file in tqdm(files):
                raw_data = self.readingFunction(file)
                raw_data.set_montage('standard_1020',verbose=50)
                channel_specific = raw_data.pick_channels(self.channels)
                badChannels = self.get_bad_channels(channel_specific)
                if len(badChannels) < len(self.channels)//2:
                    channel_specific.info['bads'] = badChannels
                    channel_specific.interpolate_bads(reset_bads=True,verbose=False)
                band_filter_data = self.butter_bandpass_filter(channel_specific.get_data())
                np.save(self.objectPath+self.preProcessedData+"{}_{}_{}.npy".format(Path(file).stem,label,counter),
                        band_filter_data)
                counter += 1


#dataLoader().loadData()
