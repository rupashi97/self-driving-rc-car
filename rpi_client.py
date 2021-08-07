import io
import socket
import struct
import time
import picamera
from drive2 import RCTest

drive = RCTest()

client_socket = socket.socket()
client_socket.connect(('192.168.137.1', 64321))
#print(client_socket.recv(1024))
#client_socket.close()

# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
    with picamera.PiCamera() as camera:

        camera.resolution = (640, 480)
        camera.framerate = 5 
        # allow the camera to warmup
        time.sleep(0.1)
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg'):
            
            # Write length of capture to stream and flush to ensure it actually gets sent
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            
            # Rewind the stream and send image data over the wire
            stream.seek(0)
            connection.write(stream.read())
           
            # Reset the stream for the next capture
            stream.seek(0)
            stream.truncate()

            direction = client_socket.recv(1024)
            drive.steer(direction)

    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()