import serial
import pygame
from pygame.locals import *
import time
import pytz
import datetime

class RCTest(object):

   def __init__(self):
       pygame.init()
       pygame.display.set_mode((250, 250))
       self.ser = serial.Serial("COM3", 9600, timeout=1)
       self.send_inst = True
       self.steer()

   def steer(self):
       f = open('Control_Data.csv', 'w' )
       f.write("Command, Timestamp \n")

       while self.send_inst:
           for event in pygame.event.get():
               if event.type == KEYDOWN:
                   key_input = pygame.key.get_pressed()

                   # complex orders
                   #  if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                   #     print("Forward Right")
                   #     self.ser.write(chr(6).encode())
                   #     f.write("Forward Right" + ",")

                   # elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                   #     print("Forward Left")
                   #     self.ser.write(chr(7).encode())
                   #     f.write("Forward Left" + ",")

                   # elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                   #     print("Reverse Right")
                   #     self.ser.write(chr(8).encode())
                   #     f.write("Reverse Right" + ",")

                   # elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                   #     print("Reverse Left")
                   #     self.ser.write(chr(9).encode())
                   #     f.write("Reverse Left" + ",") 

                   # simple orders
                   if key_input[pygame.K_UP]:
                       print("Forward")
                       self.ser.write(chr(1).encode())
                       f.write("Forward" + ",")

                   elif key_input[pygame.K_DOWN]:
                       print("Reverse")
                       self.ser.write(chr(2).encode())
                       f.write("Reverse" + ",")

                   elif key_input[pygame.K_RIGHT]:
                       print("Right")
                       self.ser.write(chr(6).encode())
                       f.write("Right" + ",")

                   elif key_input[pygame.K_LEFT]:
                       print("Left")
                       self.ser.write(chr(7).encode())
                       f.write("Left" + ",")

                   # exit
                   elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                       print("Exit")
                       self.send_inst = False
                       self.ser.write(chr(0).encode())
                       self.ser.close()
                       pygame.quit()
                       break
                   
                   utc_time = datetime.datetime.now(pytz.utc)
                   local_time = (utc_time.astimezone(pytz.timezone('Asia/Calcutta')))
                   date = (str(local_time)).split()[0]

                   f.write(str(local_time) + "\n")


               elif event.type == pygame.KEYUP:
                   self.ser.write(chr(0).encode())
                   f.write("Idle" + ",")
                   utc_time = datetime.datetime.now(pytz.utc)
                   local_time = (utc_time.astimezone(pytz.timezone('Asia/Calcutta')))
                   date = (str(local_time)).split()[0]

                   f.write(str(local_time) + "\n")
       f.close()

if __name__ == '__main__':
   RCTest()