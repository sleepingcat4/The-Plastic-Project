import sys
from PIL import Image as im
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import torch

# importing datasets
from keras.models import Sequential
from keras.utils import to_categorical, plot_model               #     
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D    # the actual layers that will go into the cnn
from sklearn.model_selection import train_test_split



# dataset 1

images = []
labels = []
img_res = [200, 200]
categories = ['Compost', 'Recycle', 'Trash']
weird_count = 0
count = 0
for i in range(len(categories)):
    for filename in os.listdir(os.getcwd() + '/ml-data/' + categories[i] + '/'):
        image = im.open(os.getcwd() + '/ml-data/' + categories[i] + '/' + filename)
        np_img = np.array(image.resize((img_res[1], img_res[0])))
        if np_img.shape == (200, 200, 3):
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
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# some preprocessing on the data

# normalizing the data to be from 0-1 instead 1-255
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# format the target data to match the output data of the cnn
# so you can compare the two
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
cnn.add(Dense(units=3, activation='softmax'))          # units is 3 because one for each number from 1-3

# print out a summary of the neural network 
print(cnn.summary())
print("\n")

# compiles the cnn with the chosen optimizer, loss metric, and metric to check
# sort of puts everything else in place
cnn.compile(optimizer='adam',                   
            loss ='categorical_crossentropy',   
            metrics=['accuracy'])               

# actually running the cnn and fitting/training neural network on the data
# batch size = after 64 samples, make a small adjustment
# epoch = every time all the data is run through, make a big adjustment

cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# evalutating the loss/accuracy of the model on the test set
loss, accuracy = cnn.evaluate(X_test, y_test)
print("loss / accuracy:")
print(loss)
print(accuracy)

# print out the prediction for the first sample in the test set
predictions = cnn.predict(X_test)
print(y_test[0]) 

print("testing some samples:")

for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability:.10%}')

print("actual value:")

print(y_test[0])

# visualize 24 random results
sns.set(font_scale=1)
index = np.random.choice(np.arange(len(X_test)), 24, replace=False)    # pick 24 random smamples
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16,9))           # set dimensions
for item in zip(axes.ravel(), X_test[index], y_test[index], predictions[index]):          # put each sample into a "slot" in the table
    axes, image, target, predict = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    print(target)
    axes.set_title("label: " + categories[np.argmax(target)] + '''
    predicted: ''' + categories[np.argmax(predict)])
plt.tight_layout()
plt.show()                      # prints it out
