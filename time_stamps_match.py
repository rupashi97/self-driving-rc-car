
import cv2
import matplotlib.pyplot as plt
import pytz
import datetime
import numpy as np 
from io import StringIO


image_stamps = np.genfromtxt('Data/T16_Image_Stamps.csv', delimiter = ',', dtype = str)
dir_stamps = np.genfromtxt('Data/T16_Direction_Stamps.csv', delimiter = ',', dtype = str)
image_stamps = image_stamps[1:,:]
dir_stamps = dir_stamps[1:,:]

print(image_stamps.shape, dir_stamps.shape)
#print(dir_stamps)

f = open('Image_Command.csv', 'a' )
f.write("Image, Command \n")
index = 0


for row in image_stamps:

	time1 = row[1].split('+')[0]
	time1 = datetime.datetime.strptime(time1, '%Y-%m-%d %H:%M:%S.%f')	
	small = []	
	

	for i in range(index, len(dir_stamps)):

		rowd = dir_stamps[i]
		if(rowd[1] == ''):
			continue
		time2 = rowd[1].split('+')[0]
		time2 = datetime.datetime.strptime(time2, '%Y-%m-%d %H:%M:%S.%f')

		
		#print(type(time1), type(time2))

		if(small!= []):
			if(abs(time1-time2) > small):
				index = i
				break
			else:
				small = abs(time1-time2)

		else:
			small = abs(time1 - time2)

		img_name = row[0]
		
		
		pass


	print(img_name, rowd[0])	
	f.write(img_name + "," + rowd[0] + "\n")
	
f.close()
