# 0D tensor : Scalars

import numpy as np
x = np.array(12)
print(x)
print(x.ndim)

# 1D Tensor: Vector

feature_one = np.array([1,2,3,4,5])
print(feature_one)
print(feature_one.ndim)


# 2D Tensor: Vector

matrices = np.array([[1,2,3,4,5],
                    [6,7,4,2,6]])
print(matrices)
print(matrices.ndim)

# 3D tensors and higher-dimensional tensors

# Packing the matrices into a new array; yields 3D tensors

x = np.array([[[1,2,3],
               [5,3,2],
               [5,5,3]],
              [[0,2,3],
               [4,3,2],
               [4,3,4]]])

print(x)
print(x.ndim)

# Key attributes of the Tensor

# Number of axes
# Shape
# Data type; float32, unit8, float64


# Display an image
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]

print(train_images.ndim)
import matplotlib.pyplot as plt

plt.imshow(digit, cmap = plt.cm.binary)#displays data as an image
plt.show()