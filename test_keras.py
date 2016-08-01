'''
@author: Nitish Bhardwaj (nitish11)
Keras Testing model for  prediction of driver's state
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import os
import cv2
import numpy as np
import pandas as pd


submission = pd.read_csv("sample_submission.csv")
cols = submission.columns[1:]
print cols

# input image dimensions
img_rows, img_cols = 60, 80
img_channels = 3
nb_classes =10

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.load_weights('my_model_weights.h5')


dirname = "dataset/test"    

filenames = os.listdir(dirname)

for f in filenames:
    print("reading",f)
    test_image = [cv2.resize((cv2.imread(os.path.join(dirname,f))),(img_cols,img_rows))]
    test_image = np.transpose(test_image, [0,3,1,2]).astype('float32')[:,:,:,:]
    prediction  = model.predict(test_image).round(2)
    # prediction = np.random.sample(10)
    submission.loc[submission["img"] == f, cols] = prediction


submission.to_csv("submissions_model.csv",index=False)