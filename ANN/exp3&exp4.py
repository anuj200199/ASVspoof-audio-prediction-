# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:40:11 2019

@author: Anuj
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
#training set
x=pd.read_csv('train/lfcc/X_balcony.csv',header=None)
y=pd.read_csv('train/lfcc/y_balcony.csv',header=None)


"""
#balcony
x1=x1.iloc[395252:511354,:]
y1=y1.iloc[395252:511354,:]
"""
"""
#office
x1=x1.iloc[:395252,:]
y1=y1.iloc[:395252,:]
"""
""" canteen
x1=x1.iloc[511354:595734,:]
y1=y1.iloc[511354:595734,:]
"""
""" home
x1=x1.iloc[595734:,:]
y1=y1.iloc[595734:,:]
"""

X_train=np.concatenate((x,x2,x1,x3))
y_train=np.concatenate((y,y2,y1,y3))
y_train=y_train[:,0:1]
X_train=X_train[:,:]
y_train=y_train[:,:]
lb=LabelEncoder()
y_train=lb.fit_transform(y_train)

X_train,x_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.0,random_state=1)


classifier=Sequential()
classifier.add(Dense(units =39 , kernel_initializer="uniform", activation = 'softmax', input_dim = 13))
classifier.add(Dense(units = 20, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 10, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 1, kernel_initializer="uniform", activation = 'sigmoid'))
optimizer=optimizers.RMSprop(lr=0.1,epsilon=None, decay=0.02)#, amsgrad=False)
# Compiling the ANN
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 64, epochs = 20)


index=pd.read_csv('protocol\protocol_V2\ASVspoof2017_V2_eval.trl.csv',delimiter=' ',header=None)
X=pd.read_csv('eval-sep/lfcc/X_balcony.csv',header=None)
Y=pd.read_csv('eval-sep/lfcc/y_balcony.csv',header=None)
"""office
X_test=np.concatenate((X,X1))
y_test=np.concatenate((Y,Y1))

"""
"""balcony
X1=X1.iloc[:65000,:]
Y1=Y1.iloc[:65000,:]
"""
"""canteen
X1=X1.iloc[65000:116000,:]
Y1=Y1.iloc[65000:116000,:]
"""
"""home
X1=X1.iloc[116000:1462500,:]
Y1=Y1.iloc[116000:1462500,:]
"""

X_test=np.concatenate((X,X2,X1,X3))
y_test=np.concatenate((Y,Y2,Y1,Y3))
X_test=X_test[:,:]
y_test=y_test[:,:]

j=y_test[:,1:2]

a=j[0]
b=[]
for i in range(0,len(j)):
    if (a!=j[i]):
        b.append(a)
        a=j[i]

y_test=y_test[:,0:1]

y_test=lb.transform(y_test)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
y_pred=np.concatenate((y_pred,j),axis=1)
yy = [];

gen=np.zeros(len(b))
spo=np.zeros(len(b))
z=0
for i in range(y_pred.shape[0]):
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

res=[]
for i in range(result.shape[1]):
    if (result[0][i]>result[1][i]):
        res.append(0)
    else:
        res.append(1)

file=[]
for i in range(len(b)):      
    file.append(b[i][0][1:])


result1=np.vstack((res,file))

actual=[]
d=[]
z=0
genu=0;
for i in range(0,len(index)):
    a=int(result1[1][z])
    m=int(index[0][i][3:-4])
    if (a==m):
        d.append(index[0][i])
        actual.append(index[1][i])
        z=z+1;

for  i in range(0,len(index)):
    a=int(result1[1][z])
    m=int(index[0][i][3:-4])
    if (a==m):
        d.append(index[0][i])
        actual.append(index[1][i])
        z=z+1;
        
actual1=np.vstack((actual,file))
y_actual=lb.transform(actual)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_actual,res)
acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_actual,res)
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
    
