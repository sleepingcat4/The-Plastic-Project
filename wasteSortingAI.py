from PIL import Image as im
import numpy as np
import torch
from matplotlib import pyplot as plt

def unpickle(file):
    import numpy as np
    with open(file, 'rb') as fo:
        data = np.load(file)
    x, y = data['x'], data['y']
    return x, y

filename = "C:\\Users\\horizon\\OneDrive\\Documents\\important\\projects\\thePlasticProject\\The-Plastic-Project\\ml-data\\Dataset2\\recycled_32_train.npz"

images, labels = unpickle(filename)

data = np.load(filename)
x_train, y_train = data['x'], data['y']
data.close()

images = images / 255.0
images = torch.tensor(images, dtype=torch.float32)
images = np.array(images)

print()
# np.rollaxis(x_train[0],0,3)
# np.transpose(x_train[0] / 255.0, (1, 2, 0))
examplefile = im.fromarray(np.transpose(images[0], (1, 2, 0)), "RGB")
examplefile.show()

