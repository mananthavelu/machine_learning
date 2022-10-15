# Hello world to deep learning practice
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# load the training and testing datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pre-process the input datasets
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Categorical encoding of the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build network / topology
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28, )))
network.add(layers.Dense(10, activation = 'softmax'))

# Compile the network
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics= ['accuracy'])
#Train the models
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Data exploration

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_images.shape)

""" 
print(type(mnist))

print(f"train_images dimension is {train_images.ndim}")
print(f"train_images dimension is {train_labels.ndim}")
print(f"train_images dimension is {test_images.ndim}")
print(f"train_images dimension is {test_labels.ndim}")
print(f"train_images type is is {type(train_images)}")

""" 
