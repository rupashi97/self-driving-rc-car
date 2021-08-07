import serial
import pygame
from pygame.locals import *
import time
import pytz
import datetime
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
#17 - left , 22 - right, 27 - forward
GPIO.setup(17, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.output(17, GPIO.HIGH)
GPIO.output(22, GPIO.HIGH) 
GPIO.output(27, GPIO.HIGH)

class RCTest(object):
    
  def __init__(self):
    pygame.init()
    pygame.display.set_mode((250, 250))
    self.steer()
 
  def steer(self):
    f = open('Direction_Stamps.csv', 'w' )
    f.write("Command, Timestamp \n")

    while 1:
      for event in pygame.event.get():
        if event.type == KEYDOWN:
          key_input = pygame.key.get_pressed()

          utc_time = datetime.datetime.now(pytz.utc)
          local_time = (utc_time.astimezone(pytz.timezone('Asia/Calcutta')))
          date = (str(local_time)).split()[0]
          f.write(str(local_time) + "\n")
        
          if key_input[pygame.K_RIGHT]:
            f.write("Right" + ",")
            GPIO.output(27, GPIO.LOW)
            GPIO.output(22, GPIO.LOW)
            time.sleep(50.0/1000.0)
            GPIO.output(27, GPIO.HIGH)  
            GPIO.output(22, GPIO.HIGH)            

          elif key_input[pygame.K_LEFT]:
            f.write("Left" + ",")
            GPIO.output(27, GPIO.LOW)
            GPIO.output(17, GPIO.LOW)
            time.sleep(50.0/1000.0)
            GPIO.output(27, GPIO.HIGH)
            GPIO.output(17, GPIO.HIGH)

          elif key_input[pygame.K_UP]:
            f.write("Forward" + ",")
            GPIO.output(27, GPIO.LOW)
            time.sleep(50.0/1000.0)
            GPIO.output(27, GPIO.HIGH)      
         
          elif key_input[pygame.K_x] or key_input[pygame.K_q]:
            print("Exit")
            pygame.quit()
            break          

        elif event.type == pygame.KEYUP:
          f.write("Idle" + ",")
          utc_time = datetime.datetime.now(pytz.utc)
          local_time = (utc_time.astimezone(pytz.timezone('Asia/Calcutta')))
          date = (str(local_time)).split()[0]
          f.write(str(local_time) + "\n")

    f.close()
    GPIO.cleanup()

if __name__ == '__main__':
   RCTest()