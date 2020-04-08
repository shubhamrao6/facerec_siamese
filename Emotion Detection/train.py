import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

num_classes = 6
img_rows, img_cols = 56, 46
batch_size = 16

train_data_dir = './datasets/train'
validation_data_dir = './datasets/validation'

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 30,
        shear_range = 0.3,
        zoom_range = 0.3,
        width_shift_range = 0.4,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'grayscale',
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
        )

from model import model
#print(model.summary())

nb_train_samples = 28273
nb_validation_samples = 3534
epochs = 30

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
