import io
import socket
import sys
import struct
import serial
import time
import pytz
import datetime

# Server socket intialization
server_socket = socket.socket()
server_socket.bind(('192.168.137.1', 59722))
print("Socket created and binded.")
#s.getsockname()
#print("Socket bound to", s.getsockname())
server_socket.listen(1)
print("Socket is listening.")
connection, add = server_socket.accept()
print("Got connection from ", add)
output =  "Thanks for connection."
connection.sendall(output.encode('utf-8'))

# Arduino control
ser = serial.Serial("COM3", 9600, timeout=1)

try:
    while True:
      f = open('Control_Data.csv', 'w')
      f.write("Command, Timestamp \n")
      output = connection.recv(1024)
      output = output.decode("utf-8")

      if(len(output)==0):
        ser.write(chr(0).encode())
        ser.close()
        break
      
      print(output)

      if output=="right":
        #print("Forward Right")
        ser.write(chr(6).encode())
        f.write("Forward Right" + ",")

      elif output=="left":
        #print("Forward Left")
        ser.write(chr(7).encode())
        f.write("Forward Left" + ",")

      elif output=="forward":
        #print("Forward")
        ser.write(chr(1).encode())
        f.write("Forward" + ",")

      utc_time = datetime.datetime.now(pytz.utc)
      local_time = (utc_time.astimezone(pytz.timezone('Asia/Calcutta')))
      date = (str(local_time)).split()[0]
      f.write(str(local_time) + "\n")
        
finally:
  f.close()
  connection.close()
  server_socket.close()