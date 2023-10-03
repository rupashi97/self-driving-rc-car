# Self-Driving RC Car

This project is a part of my bachelor's thesis, where we tackled the complex problem of building autonomous vehicles by creating a small-scale solution. Utilizing Convolutional Neural Networks (CNNs), we predicted the car's directional movement based on real-time track images. We captured these images using a Raspberry Pi and a Pi camera module and communicated with a local server through TCP/IP socket programming. F

### Here's the setup of the RC car:

<img src="https://user-images.githubusercontent.com/28508641/132827867-7a36bdab-b07b-48af-a468-92ba462fd07b.JPG" width="600" height="500">

### Track set up 

<img src="https://github.com/rupashi97/self-driving-rc-car/assets/28508641/b12512e0-bd87-434a-b477-d321e4d67acb" width="700" height="600">

## Dataset

MAC0460 Self driving dataset available on [Kaggle](https://www.kaggle.com/c/mac0460-self-driving) is used for training the classifier.

## Model Training

The training involves multiple steps:
1. **[time_stamps_match](https://github.com/rupashi97/self-driving-rc-car/blob/master/time_stamps_match.py)**: To synchronize Image_Stamps and Direction_Stamps, generating `Image_Command.csv`.
2. **[training_file_gen.ipynb](https://github.com/rupashi97/self-driving-rc-car/blob/master/notebooks/training_file_gen.ipynb)**: To create `Training_Data` and `Training_Labels` from `Image_Command.csv`.

### Models
- **ANN**: [neural_network_simulation](https://github.com/rupashi97/self-driving-rc-car/blob/master/notebooks/neural_network_simulation.ipynb) from scratch.
- **CNN**: [RC_CNN.py](https://github.com/rupashi97/self-driving-rc-car/blob/master/RC_CNN.py) (Training accuracy up to 97.64%)

## For manual Data labeling
- **Driving with Arduino and Keyboard**: [rc_car_controller](https://github.com/rupashi97/self-driving-rc-car/blob/master/rc_car_controller.py)
- **Driving with Raspberry Pi and Keyboard**: [rpi-car-controller](https://github.com/rupashi97/self-driving-rc-car/blob/master/rpi-car-controller.py)

## For Testing/Driving the car using Model Predictions
1. **[rc_rpi_controller](https://github.com/rupashi97/self-driving-rc-car/blob/master/rc_rpi_controller.py)**: Arduino-based socket connection for car commands.
2. **[drive.py](https://github.com/rupashi97/self-driving-rc-car/blob/master/drive2.py)**: RPI program to actuate car based on model's predictions.


For in-depth information and model training/testing results, please refer to our [project report](https://github.com/rupashi97/self-driving-rc-car/blob/master/Major_Project_Report.pdf).
