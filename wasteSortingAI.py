import sys
from PIL import Image as im
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import torch

# importing datasets
import keras as k
from keras.models import Sequential
from keras.utils import to_categorical, plot_model               #     
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D    # the actual layers that will go into the cnn
from sklearn.model_selection import train_test_split

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# dataset 1

images = []
labels = []
img_res = [100, 100]
categories = os.listdir(os.getcwd() + "/ml-data")
weird_count = 0
count = 0
for i in range(len(categories)):
    for filename in os.listdir(os.getcwd() + '/ml-data/' + categories[i] + '/'):
        image = im.open(os.getcwd() + '/ml-data/' + categories[i] + '/' + filename)
        np_img = np.array(image.resize((img_res[1], img_res[0])))
        if np_img.shape == (img_res[0], img_res[1], 3):
            images.append(np_img)
            labels.append(i)
            count += 1
print(count)
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.75, random_state=1)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# some preprocessing on the data

# normalizing the data to be from 0-1 instead 1-255
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# format the target data to match the output data of the cnn
# so you can compare the two
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""

Random oversampling


# reshape data into 2d arrays so it can be processed
nsamples, nx, ny, d3 = X_train.shape
X_train = X_train.reshape((nsamples,d3*nx*ny))

# Randomly oversample the minority class (garbage)
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)

# resize data back into original shape
nsamples = X_train_ros.shape[0]
X_train_ros = X_train_ros.reshape((nsamples, nx, ny, d3))

X_train = X_train_ros
y_train = y_train_ros

"""

# creates the framework for the neural network
cnn = Sequential()

# adding convolutional and pooling layers
cnn.add(Conv2D(filters = 64,            
                kernel_size = (3,3),    
                activation = 'relu',    
                input_shape=(img_res[0], img_res[1], 3)))
# doing multiple layers in order to not lose a lot of data by pooling too large
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(filters = 128, kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

# flattens the layering output into a 1D array
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))            # units = how many outputs
cnn.add(Dense(units=len(categories), activation='softmax'))  


# print out a summary of the neural network 
print(cnn.summary())
print("\n")

# compiles the cnn with the chosen optimizer, loss metric, and metric to check
# sort of puts everything else in place
cnn.compile(optimizer='adam',                   
            loss ='categorical_crossentropy',   
            metrics=['accuracy', k.metrics.AUC()])               

# actually running the cnn and fitting/training neural network on the data
# batch size = after 64 samples, make a small adjustment
# epoch = every time all the data is run through, make a big adjustment

print('labels info')
print(y_train[0])


history = cnn.fit(X_train, y_train, epochs=15, batch_size=8, validation_split=0.1)

# evalutating the loss/accuracy of the model on the test set
loss, accuracy, auc = cnn.evaluate(X_test, y_test)
print("loss / accuracy:")
print(loss)
print(accuracy)

acc = history.history['accuracy'] # get history report of the model

val_acc = history.history['val_accuracy'] # get history of the validation set

loss = history.history['loss'] #get the history of the lossses recorded on the train set
val_loss = history.history['val_loss'] #get the history of the lossses recorded on the validation set

plt.figure(figsize=(8, 8)) # set figure size for the plot generated


plt.plot(acc, label='Training Accuracy') #plot accuracy curve for each train run
plt.plot(val_acc, label='Validation Accuracy') #plot accuracy curve for each validation run

plt.legend(loc='lower right')
plt.ylabel('Accuracy') #label name for y axis
plt.ylim([min(plt.ylim()), 1]) #set limit for y axis
plt.title('Training and Validation Accuracy') #set title for the plot
plt.show()