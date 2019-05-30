import numpy as np
from random import randint
import random
import cv2
import time
import os

from keras.callbacks import TensorBoard

from helper import load_fruit_dataset

######################################################
######                 LOAD DATA                ######
######################################################

dataset_path = 'fruits-360' # replace this line by the dataset path

assert dataset_path is not None

train_set, test_set, class_idx = load_fruit_dataset(dataset_path, shuffle=True)

######################################################
######                CHECK DATA                ######
######################################################

from helper import load_image, show_image

# for class_id, img_path in train_set[:10]:
#     img = load_image(img_path)
#     print(img.shape)
#     show_image(img, size=(300, 300))

######################################################
######                VARIABLES                 ######
######################################################

experiment     = time.strftime("%Y%m%d%H%M%S")
nb_classes     = len(class_idx.keys())
input_shape    = (100, 100, 3)
batchsize      = 32
nb_steps_train = train_set.shape[0] // batchsize
nbrepoch       = 5

######################################################
######             INPUTGENERATOR               ######
######################################################
from input_generator import InputGenerator

input_train = InputGenerator(train_set, nb_classes)
assert input_train is not None

######################################################
######                  MODEL                   ######
######################################################
from helper import compile_model
from tiny_resnet import TinyResNet

model = TinyResNet(input_shape, nb_classes)
assert model       is not None

model.summary()

model = compile_model(model)

######################################################
######              MODEL TRAINING              ######
######################################################

directory = "graph/" + experiment
os.mkdir(directory)

tensorboard = TensorBoard(
    log_dir=directory, histogram_freq=0,
    write_graph=True, write_images=True
)

h = model.fit_generator(
    input_train.input_generator(batchsize=batchsize),
    steps_per_epoch=nb_steps_train,
    epochs=nbrepoch,
    verbose=1,
    callbacks=[tensorboard],
    # validation_data=input_validation.input_generator(batchsize=batchsize),
    # validation_steps=nb_steps_test,
    # max_queue_size=10, shuffle=True, initial_epoch=0,
)
