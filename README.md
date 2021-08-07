
Small scale implementation of a self driving car using Convolutional Neural Networks to predict the direction of motion of a car using track images.

## Our dataset
Training_Data, Training_Labels : R 512, L 468, F 728
Training_Data1, Training_Labels1 : R, L, F = 468

## For TRAINING:
time_stamps_match - match Image_Stamps and Direction_Stamps to get Image_Command.csv
training_file_gen.ipynb - generate Training_Data and Training_Labels from Image_Command.csv


## ANN-
neural_network_simulation (coding from scratch)
who, wih - using arduino (80% acc online dataset)
who2, wih2 - using rpi (55% acc our dataset)

nn_model (neural network using keras)


## For TRAINING:
rc_car_controller - driving car using arduino and keyboard [pygame]
rpi-car-controller - driving car using rpi and keyboard 

## For TESTING:
rc_rpi_controller - Arduino based - car command using socket connection
drive.py - (This program is in RPI) - drives car on getting command from NN predictor


online_model.h5 - saving entire trained online dataset model (83% accuracy) 
 - nn_model_online_dataset3(83acc)


## CNN:
script - RC_CNN.py
CNN_model_1.h5 - trained for 5 epochs | train acc = 91% | test acc = 89.1%

script - RC_CNN.py
CNN_model_2.h5 - trained for 10 epochs | train acc = 97.64% | test acc = 92%


