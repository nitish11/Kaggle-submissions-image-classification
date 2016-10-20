'''
Author : Nitish
Date : September 13th, 2016
Description : Cifar-10 training model
Version : 1.0
'''

#loading dependencies
import cv2
import numpy as np
import os
import sys
import Image
import glob
import configparser
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D
from keras.optimizers import SGD,Adam,RMSprop,Adagrad,Adadelta,Adamax
from keras.callbacks import ModelCheckpoint, Callback



#To read the training data and create training set for keras model 
def prepare_data(no_classes,dataset_path,num_of_channels,image_width,image_height):

	print "===Setting the data ==="
	class_names = os.listdir(dataset_path)

	#Calculate the number of the images in the dataset
	N=0
	for p_name in class_names:
		i_path = dataset_path + "/"+p_name 
		i_count = len(os.listdir(i_path))
		N += i_count

	# Training Data
	X_train = np.zeros((N, num_of_channels, image_width, image_height), dtype=np.uint8)
	y_train = np.zeros((N,no_classes), dtype=np.int64)

	index = 0
	data_count =0
	data_class = 0

	#class list and Images to read from directory
	for class_index,class_name in enumerate(class_names):
		
		images_path = dataset_path + "/"+class_name 
		images_count = len(os.listdir(images_path))
		label_data = np.zeros((no_classes), dtype=np.uint8)
		data_class += 1
		data_count += images_count
		
		print "= class name",class_name
		images_filenames = glob.glob(images_path + '/*.png')
		train_images = [np.array(cv2.resize(cv2.imread(f),(image_width,image_height))) for f in images_filenames]
		train_images = np.array(train_images)/255. # Normalize data
		num_of_images = len(train_images) 
		
		train_images = np.transpose(train_images, [0, 3, 1, 2]).astype('float32')[:num_of_images, :, :, :]
		train_index = index+num_of_images 
		
		X_train[index:train_index] = train_images
		#Labelling data
		label_data[data_class-1] = 1
		y_train[index:train_index] = label_data
		
		index += num_of_images
		if data_class == no_classes:
			break

	print "Xtrain shape :- " + str(np.shape(X_train)) 
	print "Ytrain shape :- " + str(np.shape(y_train)) 
	print "==Number of classs taken into data :" + str(data_class) 

	return X_train,y_train






#main function
if __name__ == "__main__":

	
	batch_size = 1
	nb_classes = 3
	nb_epoch = 10
	data_augmentation = True

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3

	#Preparing Data
	X_train,Y_train = prepare_data(nb_classes,'./train-data',img_channels,img_rows,img_cols)
	X_val,Y_val = prepare_data(nb_classes,'./test-data',img_channels,img_rows,img_cols)

	#saves the model weights after each epoch if the validation loss decreased
	checkpointer = ModelCheckpoint(filepath="./model_weights.hdf5", verbose=1, save_best_only=True)


	#CIFAR-10 model
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',
	                        input_shape=X_train.shape[1:]))
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


	print " === Training the sample data ==="	
	model.fit(X_train, Y_train, batch_size=batch_size,nb_epoch=nb_epoch, verbose=2,validation_data=(X_val,Y_val), shuffle=True,callbacks=[checkpointer])

	print "== Training completed =="
	
