from PIL import Image as im
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import seaborn as sns

# importing datasets
from tensorflow.keras.utils import to_categorical, plot_model               #     
from tensorflow.keras.models import Sequential                              # constructor for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D    # the actual layers that will go into the cnn
from sklearn.model_selection import train_test_split



# loading dataset 2 (not working)

# def unpickle(file):
#     import numpy as np
#     with open(file, 'rb') as fo:
#         data = np.load(file)
#     x, y = data['x'], data['y']
#     return x, y

# filename = "C:\\Users\\horizon\\OneDrive\\Documents\\important\\projects\\thePlasticProject\\The-Plastic-Project\\ml-data\\Dataset2\\recycled_32_train.npz"

# images, labels = unpickle(filename)

# data = np.load(filename)
# x_train, y_train = data['x'], data['y']
# data.close()

# images = images / 255.0
# images = torch.tensor(images, dtype=torch.float32)
# images = np.array(images)

# np.rollaxis(x_train[0],0,3)
# np.transpose(x_train[0] / 255.0, (1, 2, 0))
# examplefile = im.fromarray(np.transpose(images[0], (1, 2, 0)), "RGB")
# examplefile.show()



# dataset 1

images = []
labels = []
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
for c in categories:
    for filename in os.listdir(os.getcwd() + '\\ml-data\\Dataset1\\' + c + '\\'):
        image = im.open(os.getcwd() + '\\ml-data\\Dataset1\\' + c + '\\' + filename)
        np_img = np.array(image)
        images.append(np_img)
        labels.append(c)

# examplefile = im.fromarray(images[0])
# examplefile.show()

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

# visualize 24 random samples
sns.set(font_scale=2)
index = np.random.choice(np.arange(len(X_train)), 24, replace=False)    # pick 24 random smamples
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16,9))           # set dimensions
for item in zip(axes.ravel(), X_train[index], y_train[index]):          # put each sample into a "slot" in the table
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
plt.show()                      # prints it out
