from cnn_model.smallcnn import SmallCNN
import pandas as pd 
import numpy as np 
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import random

import cv2 
import os
from PIL import Image
import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import nibabel as nib

from utils import read_nii, clahe_enhancer, cropper

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import  Conv2D,  MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K

from pylab import rcParams

if __name__ == "__main__":
    raw_data = pd.read_csv('../metadata.csv')
    raw_data = raw_data.replace('../input/covid19-ct-scans/','../',regex=True)

    # initialize
    all_points1 = []
    all_points2 = []
    cts = []
    lungs = []
    infections = []
    img_size = 224
    
    # read image data
    for i in range(0, raw_data.shape[0]):
        read_nii(raw_data.loc[i,'lung_mask'], lungs, 'lungs')
        read_nii(raw_data.loc[i,'ct_scan'], cts, 'cts') 
        read_nii(raw_data.loc[i,'infection_mask'], infections, 'infections')

    # load target data
    y_label = []
    for i in range(0, len(infections)):
        if len(np.unique(infections[i]))!=1:
            y_label.append(1)
        else:
            y_label.append(0)
            
    # convert to np array
    cts = np.array(cts).astype('uint8')
    cts = cts.reshape(len(cts), img_size,img_size,1)
    y_label = np.array(y_label)
    
    # split data
    x_train, x_valid, y_train, y_valid = train_test_split(cts, y_label, test_size = 0.3, random_state=42)
    
    #data augmentation
    aug = ImageDataGenerator( 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True, 
        fill_mode="nearest"
        )
    
    model = SmallCNN.build(img_size,img_size, 1)
    
    # prepare model
    batch_size = 32
    epochs = 50
    best_val_auc = -1

    #model checkpoint
    filepath_acc = "covid_weights_val_acc.hdf5"
    checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003), metrics=["acc"])
    
    # calculate class weights
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)
    weights=dict(enumerate(weights))

    # train model
    results = model.fit(aug.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=(x_valid, y_valid) ,
                        steps_per_epoch = len(x_train)//batch_size,
                        callbacks = [checkpoint_acc],
                        class_weight=weights)
    
    rcParams['figure.figsize'] = 10,7
    plt.grid('True')
    plt.plot(results.history['loss'], color='m')
    plt.plot(results.history['val_loss'], color='k')
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.show()
    
    rcParams['figure.figsize'] = 10,7
    plt.grid('True')
    plt.plot(results.history['acc'], color='m')
    plt.plot(results.history['val_acc'], color='k')
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.show()