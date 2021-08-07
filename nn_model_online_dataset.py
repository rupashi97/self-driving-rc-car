import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mp  
import csv 
import pandas as pd 
import math
import  cv2

# model
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
# split and other utils
from sklearn.model_selection import train_test_split

# Loading dataset
train_X = np.load("online_dataset/train_X.npy")
train_Y = np.load("online_dataset/train_y.npy")
test_X = np.load("online_dataset/test_X.npy")
test_Y = np.load("online_dataset/test_y.npy")

# Preprocessing dataset
# traning data
train_X_new = np.empty((train_X.shape[0], 45*80))
index = 0

for record in train_X:    
    sample = cv2.cvtColor(np.reshape(record, (45,80, 3)), cv2.COLOR_BGR2GRAY)
    sample = sample.reshape((1,45*80))
    sample = (np.asfarray(sample)/ 255.0) 
    if index < train_X.shape[0]:
        train_X_new[index, :] = sample
    index = index + 1

# testing data
test_X_new = np.empty((test_X.shape[0], 45*80))
index = 0

for record in test_X:    
    sample = cv2.cvtColor(np.reshape(record, (45,80, 3)), cv2.COLOR_BGR2GRAY)
    sample = sample.reshape((1,45*80))
    sample = (np.asfarray(sample)/ 255.0) 
    if index < test_X.shape[0]:
        test_X_new[index, :] = sample
    index = index + 1
    
# Correcting nomenclature
train_X_old = train_X
test_X_old = test_X
train_X = train_X_new
test_X = test_X_new
print(train_X_old.shape, test_X_old.shape, train_X.shape, test_X.shape)

# Splitting dataset into training, testing and validation
print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
print(train_X.shape, valid_X.shape, train_Y.shape, valid_Y.shape)

train_Y = to_categorical(train_Y, num_classes = 3)
valid_Y = to_categorical(valid_Y, num_classes = 3)

print(train_Y.shape, train_Y)
print(valid_Y.shape, valid_Y)

row = 45
col = 80

model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(output_dim = 64, init = 'uniform', activation = 'sigmoid', input_dim = row*col))
# Adding the second hidden layer
model.add(Dense(output_dim = 32, init = 'uniform', activation = 'sigmoid'))
# Adding the output layer
model.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

epochs = 100
learning_rate = 0.05
decay_rate = learning_rate/epochs
momentum = 0.8
sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov=False)

# Compiling the ANN
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.summary()

# Fitting the ANN to the Training set
history = model.fit(train_X, train_Y, batch_size = 5, epochs = epochs, validation_data=(valid_X, valid_Y))

# Plotting training results
print(history.history.keys())

#  "Accuracy"
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Evaluate using Test data
test_Y_ = to_categorical(test_Y)
scores = model.evaluate(test_X, test_Y_, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predicting the Test Results
pred_y = model.predict(test_X)
print(pred_y.shape,"\n", pred_y)

i=0
pred_Y = np.empty((pred_y.shape[0],1))
for rec in pred_y:
    pred_Y[i] = np.argmax(rec > 0.5)
    i = i+1    
print(pred_Y.shape)

# Confusion Matrix
test_Y = test_Y.reshape((test_Y.shape[0],1))  # to make a 2d matrix
cm = confusion_matrix(test_Y,pred_Y)
print(cm,"\n")
print("Confusion Matrix : \n")
for i in range(3):
    print(np.round(cm[i]/np.sum(cm[i])*100,2))
    
# Performance accuracy 
accuracy = ((test_Y == pred_Y)==True).sum() / len(test_Y) * 100
print("\nAccuracy = "+ str(accuracy) + "%\n")

# Save model as a .h5 file
model.save('NN_model_1.h5')