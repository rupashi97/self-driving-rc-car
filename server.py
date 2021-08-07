# importing libraries for Socket programming and Pre processing
import socket
import sys
import cv2
import pickle
import numpy as np
import struct 
import io
import matplotlib.pyplot as plt

# Keras Libraries
import keras
from keras.models import load_model
from keras.layers.convolutional import Conv2D

# Loading trained CNN model
model = load_model('CNN_model_2.h5')
print("Model loaded\n")
arr = ["forward", "left", "right"]
i = 1

# Socket intialization
server_socket = socket.socket()
print('Socket created')

server_socket.bind(('192.168.137.1', 64321))
print('Socket bind complete')
print("Socket bound to", server_socket.getsockname()[1])

server_socket.listen(10)
print('Socket now listening')

conn, addr = server_socket.accept()
# print("Got connection from ", addr)
# output =  "Thanks for connection."
# conn.sendall(output.encode('utf-8'))

# Accept a single connection and make a file-like object out of it
#connection = server_socket.accept()[0].makefile('rb')
connection = conn.makefile('rb')
print("Got connection from RPI.\n")

try:
	# Read the length of the image; if length = 0; quit

    while True:
    	image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    	if not image_len:
    		break
    	# Construct a stream to hold image data and read image data from the connection
    	image_stream = io.BytesIO()
    	image_stream.write(connection.read(image_len))
    	image_stream.seek(0)

    	data = np.fromstring(image_stream.getvalue(), dtype = np.uint8)
    	image = cv2.imdecode(data, 1)

    	rows = image.shape[0]
    	cols = image.shape[1]
    	M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    	image = cv2.warpAffine(image,M,(cols,rows))

    	cv2.imshow('Frame', image)
    	if cv2.waitKey(1) & 0xFF == ord("q"):
    		break

    	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    	# plt.imshow(img, cmap = 'gray')
    	# plt.show()

    	img = img[150:450,0:img.shape[1]]

    	img = cv2.resize(img, (80,45), interpolation = cv2.INTER_AREA)
    	#img = cv2.GaussianBlur(img, (3,3), 0)

    	img1 = img.reshape((1,45,80,1))

    	inp = (np.asfarray(img1)/ 255.0)
    	pred_y = model.predict(inp)

    	direction = np.argmax(pred_y > 0.5)

    	name =  "test_images/" + str(i) + "-" + str(arr[direction]) + ".jpg"
    	cv2.imwrite(name, img)
    	
    	direction = str(direction)
    	print(direction)
    	conn.sendall(direction.encode('utf-8'))
    	i = i+1

finally:
    connection.close()
    server_socket.close()