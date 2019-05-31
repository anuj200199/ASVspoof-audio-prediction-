# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:14:32 2019

@author: Anuj
"""
"""
    E01: 'Anechoic room'
    E02: 'Balcony 01'
    E03: 'Balcony 02'
    E04: 'Home 07'
    E05: 'Home 08'
    E06: 'Cantine'
    E07: 'Home 01'
    E08: 'Home 02'
    E09: 'Home 03'
    E10: 'Home 04'
    E11: 'Home 05'
    E12: 'Home 06'
    E13: 'Office 01'
    E14: 'Office 02'
    E15: 'Office 03'
    E16: 'Office 04'
    E17: 'Office 05'
    E18: 'Office 06'
    E19: 'Office 07'
    E20: 'Office 08'
    E21: 'Office 09'
    E22: 'Office 10'
    E23: 'Studio'
    E24: 'Analog wire 01'
    E25: 'Analog wire 02'
    E26: 'Analog wire 03'
"""
import pandas as pd 

data_t=pd.read_csv('train/lfcc_norm_all.csv',header=None)
#data_t1=pd.read_csv('train/mfcc_norm_all.csv',header=None)
#data_t2=pd.read_csv('train/imfcc_norm_all.csv',header=None)
#data_t3=pd.read_csv('train/rfcc_norm_all.csv',header=None)

protocol=pd.read_csv('protocol/protocol_V2/ASVspoof2017_V2_train.trn.csv',delimiter=' ',header=None)


X_t=data_t.iloc[:,-1:].values
#X_d=data_t1.iloc[:,:-1].values
env_train=protocol.iloc[:,4:5].values
filename_train=protocol.iloc[:,0:1].values

z=0
actual_train=[]
for i in range(0,X_t.shape[0]):
    a=int(X_t[i][0][1:])
    m=int(filename_train[z][0][3:-4])
    if (a==m):
        actual_train.append(env_train[z])
    else:
        actual_train.append(env_train[z+1])
        z=z+1;  
        
X_features_t=data_t.iloc[:,1:40].values
y_features_t=data_t.iloc[:,40:].values

import numpy as np
final_train=np.concatenate((X_features_t,actual_train),axis=1)
anechoic_t=[]#E01
balcony_t=[]#02 03
home_t=[]#07 08 09 10 11 12 4 5 
office_t=[]#13 to 22
studio_t=[]#23
analog_t=[]#24 25 26
canteen_t=[]#06
an_t=[]
st_t=[]
ba_t=[]
analo_t=[]
ca_t=[]
ho_t=[]
gen_t=[]
of_t=[]
genuine_t=[]
count=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
for i in range(0,protocol.shape[0]):
    if final_train[4][i]=='E01':
        count=count+1
        anechoic_t.append(X_features_t[i])
        an_t.append(y_features_t[i])
    elif (final_train[4][i]=='E23'):
        count1=count1+1
        studio_t.append(X_features_t[i])
        st_t.append(y_features_t[i])
    elif (final_train[4][i]=='E02' or final_train[4][i]=='E03'):
        count3=count3+1
        balcony_t.append(X_features_t[i])
        ba_t.append(y_features_t[i])
    elif (final_train[4][i]=='E24' or final_train[4][i]=='E25' or final_train[4][i]=='E26'):
        count4=count4+1
        analog_t.append(X_features_t[i])
        analo_t.append(y_features_t[i])
    elif (final_train[4][i]=='E06'):
        count5=count5+1
        canteen_t.append(X_features_t[i])
        ca_t.append(y_features_t[i])
    elif (final_train[4][i]=='E04' or final_train[4][i]=='E05' or final_train[4][i]=='E07' or final_train[4][i]=='E08' or final_train[4][i]=='E09'
          or final_train[4][i]=='E10' or final_train[4][i]=='E11' or final_train[4][i]=='E12'):
        count6=count6+1
        home_t.append(X_features_t[i])
        ho_t.append(y_features_t[i])
    elif (final_train[4][i]=='-'):
        count8=count8+1
        genuine_t.append(X_features_t[i])
        gen_t.append(y_features_t[i])
    else:
        count9=count9+1
        office_t.append(X_features_t[i])
        of_t.append(y_features_t[i])
        
  
import numpy as np


np.savetxt('train/lfcc/X_genuine.csv',genuine_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/ane.csv',anechoic_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/std.csv',studio_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/bal.csv',balcony_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/ana.csv',analog_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/can.csv',canteen_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/home.csv',home_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/off.csv',office_t,fmt='%s',delimiter=',')

np.savetxt('train/lfcc/y_genuine.csv',gen_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_ane.csv',an_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_std.csv',st_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_bal.csv',ba_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_ana.csv',analo_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_can.csv',ca_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_home.csv',ho_t,fmt='%s',delimiter=',')
np.savetxt('train/lfcc/y_off.csv',of_t,fmt='%s',delimiter=',')





