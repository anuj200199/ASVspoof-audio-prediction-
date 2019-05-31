# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:00:26 2019

@author: Anuj
"""

import pandas as pd

data=pd.read_csv('mfcc_final.csv')
data1=data.iloc[:,1:14].values
x=data.iloc[:,0:1].values
y=data.iloc[:,14:].values

rows,cols=data1.shape
import numpy as np
norm=np.mean(data1,axis=0)
norm_vec=np.tile(norm,(rows,1))

mean_sub=data1-norm_vec

stddec=np.std(mean_sub,axis=0)
stddev_vec=np.tile(stddec,(rows,1))

output=mean_sub/stddev_vec
output=np.concatenate((x,output),axis=1)
output=np.concatenate((output,y),axis=1)
import csv
with open('norm_test.csv','w') as csvfile:    
     writer = csv.writer(csvfile)
     writer.writerows(output)
    
