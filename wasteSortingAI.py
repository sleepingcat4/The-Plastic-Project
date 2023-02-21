from collections import Counter
from PIL import Image as im
import numpy as np
from matplotlib import pyplot as plt
import os
import sklearn
import keras as k
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical                 
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, RandomRotation, RandomFlip, RandomZoom, RandomContrast
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from PIL import ImageFile
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

tf.config.set_soft_device_placement(True) 
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load dataset

images = []
labels = []
img_res = [100, 100]
categories = os.listdir(os.getcwd() + "/data-v1")
weird_count = 0
count = 0

for i in tqdm(range(len(categories))):
    for filename in tqdm(os.listdir(os.getcwd() + '/data-v1/' + categories[i] + '/'), leave=False):
        image = im.open(os.getcwd() + '/data-v1/' + categories[i] + '/' + filename)
        np_img = np.array(image.resize((img_res[1], img_res[0])))
        if np_img.shape == (img_res[0], img_res[1], 3):
            images.append(np_img)
            labels.append(i)
            count += 1

# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.75, random_state=1)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# reshape data into 2d arrays so it can be processed
nsamples, nx, ny, d3 = X_train.shape
X_train = X_train.reshape((nsamples,d3*nx*ny))

# Randomly oversample minority classes
X_train_ros, y_train_ros= SMOTE().fit_resample(X_train, y_train)

# Check distribution of data across classes
print(sorted(Counter(y_train_ros).items()))

# resize data back into original shape
nsamples = X_train_ros.shape[0]
X_train_ros = X_train_ros.reshape((nsamples, nx, ny, d3))

X_train = X_train_ros
y_train = y_train_ros

# some preprocessing on the data

# normalizing the data to be from 0-1 instead 1-255
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# format the target data to match the output data of the cnn
# so you can compare the two
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# data augmentation
with tf.device('/cpu:0'):
    data_augmentation = k.Sequential([
    RandomRotation(0.2, input_shape=(img_res[0], img_res[1], 3)),
    RandomZoom(0.2, 0.2),
    ])

# creates the framework for the neural network
cnn = Sequential(
    [
    data_augmentation,
    Conv2D(filters = 64,            
                kernel_size = (3,3),    
                activation = 'relu',    
                input_shape = (img_res[0], img_res[1], 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters = 128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=len(categories), activation='softmax')
    ]
)

print(cnn.summary())


# shuffle data
X_train, y_train = sklearn.utils.shuffle(X_train, y_train)


epoch_counter = 1

# save the data to a file that can later be converted to the CoreML format
class SaveModelCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global epoch_counter
        print("Saving model...")
        self.model.save("model_epoch_" + str(epoch_counter) + ".h5")
        epoch_counter += 1
        

# actually running the cnn and fitting/training neural network on the data
# batch size = after 64 samples, make a small adjustment
# epoch = every time all the data is run through, make a big adjustment
with tf.device("/gpu:0"):
    cnn.compile(optimizer='adam', 
                loss=k.losses.CategoricalCrossentropy(),
                metrics=[k.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),'accuracy'])           
    history = cnn.fit(X_train, y_train, epochs=12, batch_size=16, validation_split=0.1, callbacks=[SaveModelCallback()])

# evalutating the loss/accuracy of the model on the test set
loss, crossent, accuracy = cnn.evaluate(X_test, y_test)
print("loss / crossentropy / accuracy:")
print(loss)
print(crossent)
print(accuracy)

acc = history.history['accuracy'] # get history report of the model

val_acc = history.history['val_accuracy'] # get history of the validation set

loss = history.history['loss'] #get the history of the lossses recorded on the train set
val_loss = history.history['val_loss'] #get the history of the lossses recorded on the validation set

y_pred = cnn.predict(X_test)

y_pred = (y_pred > 0.5) 

print(classification_report(y_test, y_pred, target_names=categories, digits=4))


# plt.figure(figsize=(8, 8)) # set figure size for the plot generated


# plt.plot(acc, label='Training Accuracy') #plot accuracy curve for each train run
# plt.plot(val_acc, label='Validation Accuracy') #plot accuracy curve for each validation run

# plt.legend(loc='lower right')
# plt.ylabel('Accuracy') #label name for y axis
# plt.ylim([min(plt.ylim()), 1]) #set limit for y axis
# plt.title('Training and Validation Accuracy') #set title for the plot
# plt.show()
