# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:31:00 2019

@author: Anuj



"""
#import library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras import callbacks


#training set(vary combinations as required in X_train)
d=pd.read_csv('train/mfcc_norm_all.csv')
data=pd.read_csv('train/lfcc_norm_all.csv')
#data1=pd.read_csv('train/rfcc_norm_train_all.csv',header=None)
#data4=pd.read_csv('train/imfcc_norm_train_all.csv',header=None)
x=d.iloc[:,1:40].values
X_tr = data.iloc[:,1:40].values
y_train = data.iloc[:,-2].values
#X_f=data1.iloc[:,1:40].values
#X_r = data4.iloc[:,1:40].values

X_train=np.concatenate((x,X_tr),axis=1)

#validation set(vary combinations as required in X)
d1=pd.read_csv('dev/mfcc_norm_all.csv',header=None)
data2=pd.read_csv('dev/lfcc_norm_all.csv')
#data3=pd.read_csv('dev/rfcc_norm_dev_all.csv',header=None)
#data5=pd.read_csv('dev/imfcc_norm_dev_all.csv',header=None)
x1=d1.iloc[:,1:40].values
X_tr1=data2.iloc[:,1:40].values
#X_f2=data3.iloc[:,1:40].values
#X_r1=data5.iloc[:,1:40].values
y=data2.iloc[:,-2]

X=np.concatenate((x1,X_tr1),axis=1)

#encoding the output as 0(genuine) and 1(spoof)
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y = labelencoder_y.transform(y)

from sklearn.cross_validation import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.0,random_state=1)
X,X_val,y,y_val=train_test_split(X,y,test_size=0.0,random_state=1)


#building the model
classifier = Sequential()
classifier.add(Dense(units =39 , kernel_initializer="uniform", activation = 'softmax', input_dim = 39))
classifier.add(Dense(units = 20, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 10, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 1, kernel_initializer="uniform", activation = 'sigmoid'))
optimizer=optimizers.RMSprop(lr=0.1,epsilon=None, decay=0.02)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X, y))

#protocol file already given in dataset showing details of each file(used to get actual value)
index=pd.read_csv('protocol\protocol_V2\ASVspoof2017_V2_eval.trl.csv',delimiter=' ',header=None)

#test set (vary combinations as required in X_test)
dat=pd.read_csv('eval/mfcc_norm_all.csv',header=None)
da=pd.read_csv('eval/lfcc_norm_all.csv',header=None)
#da1=pd.read_csv('eval/rfcc_norm_all.csv',header=None)
#da2=pd.read_csv('eval/imfcc_norm_all.csv',header=None)
c=dat.iloc[:,1:40].values
X_d=da.iloc[:,1:40].values
#X_d2=da1.iloc[:,1:40].values
#X_d4=da2.iloc[:,1:40].values

X_test=np.concatenate((c,X_d),axis=1)
y_test=da.iloc[:,-2].values

#ensoding y(already known)
y_test=labelencoder_y.transform(y_test)
j=da.iloc[:,-1:]

#counting the number of unique files
b=j[41].unique()

#predicting for all frames
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
y_pred=np.concatenate((y_pred,j),axis=1)
yy = [];


#copunting the number of genuine ans spoof predictions for all frame of each audio file
gen=np.zeros(len(b))
spo=np.zeros(len(b))
z=0
for i in range(len(y_pred)):
    if(y_pred[i][1]==b[z]):
        if y_pred[i][0] is False:
            gen[z]=gen[z]+1;
        else:
            spo[z]=spo[z]+1;
    else:
        if y_pred[i][0] is False:
            gen[z+1]=gen[z+1]+1;
        else:
            spo[z+1]=spo[z+1]+1;
        z=z+1;
    
        
result=np.vstack((gen,spo))

#predicted labelling values of each audio file (0-genuine and 1-spoof)
res=[]
for i in range(result.shape[1]):
    if (result[0][i]>result[1][i]):
        res.append(0)
    else:
        res.append(1)

        
file=[]
for i in range(len(b)):      
    file.append(b[i][1:])


result1=np.vstack((res,file))

#getting the actual details regarding each file from train.trl inside protocol already given stored in index
actual=[]
z=0
genu=0;
for i in range(0,len(index)):
    a=int(result1[1][z])
    m=int(index[0][i][3:-4])
    if (a==m):
        actual.append(index[1][i])#index[1][i] gest actual labellingo of file from train.trl
        z=z+1;
        

actual1=np.vstack((actual,file))

y_actual=labelencoder_y.transform(actual)

#getting accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(res,y_actual)
acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])


#roc_curve and eer calculation
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(res,y_actual)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

from scipy.optimize import brentq
from scipy.interpolate import interp1d
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
