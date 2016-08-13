#from __future__ import absolute_import
#from ..utils.data_utils import get_file
import numpy as np
import os
import cv2


def load_data():

    classNames = os.listdir('train') # Also the number of folders

    nb_train_samples = 0

    for className in classNames:
        nb_train_samples +=len(os.listdir('train/'+className))
    
    classNames.sort()
    
    X_train = np.zeros((nb_train_samples, 3, 24, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,len(classNames)), dtype="uint8")

    #for i in range(1, 6):
    nb_samples_currently = 0
    for i in xrange(len(classNames)):
        nb_samples_current = len(os.listdir('train/'+classNames[i]))
        print nb_samples_currently,nb_samples_currently+nb_samples_current
        fpath = os.path.join('train', classNames[i])
        #data = data.reshape(data.shape[0], 3, 32, 32)
        data, labels = load_batch(fpath,i)
        print data.shape
        X_train[nb_samples_currently:nb_samples_currently+nb_samples_current, :, :, :] = data
        y_train[nb_samples_currently:nb_samples_currently+nb_samples_current] = labels
        nb_samples_currently += nb_samples_current
        
    return (X_train, y_train)
    
    
def load_batch(fpath, label):
    print fpath
    filenames = os.listdir(fpath)
    filenames.sort(key=len)    
    filenames.sort()
    
    data = []    
    labels = []
    for filename in filenames:
        img = cv2.imread(fpath+'/'+filename)
        img = cv2.resize(img,(32,24))
        data.append(img)
        label_array= np.repeat(0,10)
        np.put(label_array,label,1)
        labels.append(label_array)
    data = np.array(data)
    data = data.reshape(data.shape[0], 3, 24, 32)
    labels = np.array(labels)
    return data,labels        