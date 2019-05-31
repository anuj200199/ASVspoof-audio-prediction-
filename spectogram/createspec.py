# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:18:55 2019

@author: Anuj
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import librosa

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
        samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
        # cols for windowing
        cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)   
        
        
def logscale_spec(spec, sr=44100, factor=20.):
        timebins, freqbins = np.shape(spec)

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):        
            if i == len(scale)-1:
                newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
            else:        
                newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

        return newspec, freqs


        
def plotstft(audiopath,filepath, binsize=2**8,colormap="jet"):
        samples, samplerate = librosa.load(audiopath,sr=44100)
        x=librosa.core.get_duration(samples,samplerate)
        t1 = np.linspace(0, x, samplerate * x)
        imfs=EMD(samples)
        dec=imfs.decompose()
        samples=dec[0]+dec[1]
    
        s = stft(samples, binsize)
            
        sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
            
        ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

        timebins, freqbins = np.shape(ims)
            
        print("timebins: ", timebins)
        print("freqbins: ", freqbins)
            
        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
            
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
            #plt.ylim(0,47)
            
        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.savefig(filepath, bbox_inches="tight")
            
        plt.clf()
            
            
import os
import pandas as pd
filename=pd.read_csv('protocol/protocol_V2/ASVspoof2017_V2_train.trn.csv',header=None,delimiter=' ')
for i in range(900,1000):
    if (filename[1][i]=='genuine'):
        ims = plotstft(audiopath=os.path.join("train/ASVspoof2017_V2_train",filename[0][i]),filepath=os.path.join("train/genimfcomb",'genuine('+str(i)+').jpg'))
    else:
        ims = plotstft(audiopath=os.path.join("train/ASVspoof2017_V2_train",filename[0][i]),filepath=os.path.join("train/spoimfcomb",'spoof('+str(i)+').jpg'))
    
   
