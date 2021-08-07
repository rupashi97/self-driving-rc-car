# For TESTING using keras weights
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import serial
import datetime
import pytz
import numpy as np
from drive2 import RCTest
import RPi.GPIO as GPIO
from neural_testing import neuralNetwork
import keras
from keras.models import load_model

model = load_model('online_model.h5')
label = ["forward", "left", "right"]

drive = RCTest()

# initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 5
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

  # grab the raw NumPy array, and rotate image
  image = frame.array
  rows = image.shape[0]
  cols = image.shape[1]
  M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
  image = cv2.warpAffine(image,M,(cols,rows))

  # display the frame
  cv2.imshow("Frame", image)

  # Pre process image and send to neural network
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  img = img[200:img.shape[0],0:img.shape[1]]
  img = cv2.resize(img, (45,80), interpolation = cv2.INTER_AREA)
  img = cv2.GaussianBlur(img, (5,5), 0)
  #img = cv2.fastNlMeansDenoising(img,None,4,7,21)
  #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

  #img = img.reshape((1,img.size))
  input = (np.asfarray(img)/ 255.0)
  pred_y = model.predict(input)
  pred_Y= np.argmax(pred_y > 0.5)
  drive.steer(pred_Y)

  # clear the stream in preparation for the next frame
  rawCapture.truncate(0)

  # Press q to exit
  if cv2.waitKey(1) & 0xFF == ord("q"):
         break