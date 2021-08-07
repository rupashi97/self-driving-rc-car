import time
import pytz
import RPi.GPIO as GPIO

class RCTest(object):

  def __init__(self):
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    #17 - left , 22 - right, 27 - forward
    GPIO.setup(17, GPIO.OUT)
    GPIO.setup(22, GPIO.OUT)
    GPIO.setup(27, GPIO.OUT)

    GPIO.output(17, GPIO.HIGH)
    GPIO.output(22, GPIO.HIGH) 
    GPIO.output(27, GPIO.HIGH)
    
    self.steer()

  def __del__(self):
    GPIO.cleanup()

  def steer(self, dir):
    if dir==2:
      print("Right\n")
      GPIO.output(27, GPIO.LOW)
      GPIO.output(22, GPIO.LOW)
      time.sleep(50.0/1000.0)
      GPIO.output(27, GPIO.HIGH)  
      GPIO.output(22, GPIO.HIGH)
  
    elif dir==1:
      print("Left\n")
      GPIO.output(27, GPIO.LOW)
      GPIO.output(17, GPIO.LOW)
      time.sleep(50.0/1000.0)
      GPIO.output(27, GPIO.HIGH)
      GPIO.output(17, GPIO.HIGH)

    elif dir==0:
      print("Forward\n")
      GPIO.output(27, GPIO.LOW)
      time.sleep(50.0/1000.0)
      GPIO.output(27, GPIO.HIGH)
    