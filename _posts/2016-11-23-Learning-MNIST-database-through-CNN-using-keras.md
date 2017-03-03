---
layout: post
title: Learning MNIST database through CNN using Keras
categories: [ComputerVision]
author: ISR
---

Create a Convolutional Neural Network using Keras to learn MNIST database. The CNN provides an accuracy of around 99% on th MNIST database. Keras is a python library for deep learning.
<!--more-->


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```


```python
import keras
from keras.datasets import mnist
```

    Using TensorFlow backend.



```python
(X_train,y_train),(X_test,y_test) = mnist.load_data()
```

    Downloading data from https://s3.amazonaws.com/img-datasets/mnist.pkl.gz



```python
plt.imshow(X_train[5])
plt.show()
```


![png](output_3_0.png)


### Preprocessing
* reshaping input according to keras requirement for tensorflow background.
    * for 2d-convnet in keras input = [numInstances, height, width, 1] ---- i.e 1 for grayscale images
* normalizing input between [0,1]
* one-hot encoding for the target variables



```python
from keras.utils.np_utils import to_categorical
```


```python
lenTrain,height,width = X_train.shape
lenTest,height,width = X_test.shape
X_train = X_train.reshape(lenTrain,height,width,1).astype("float32")
X_test = X_test.reshape(lenTest,height,width,1).astype("float32")
```


```python
X_train /= 255
X_test /=255

```


```python
numClasses = 10
y_train = to_categorical(y_train,numClasses)
```


```python
y_test = to_categorical(y_test,numClasses)
```

## Creating the Model


```python
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout 
```

* In Keras, the model is initialized as a Sequential Object. 
* The whole CNN architecture can be created by adding layers through the add function.
* The model then is compiled, using the compile function.
* The main inputs to the compile function are :
    * cost function
    * optimizer 
* Then the compiled model is trained on the training data set through fit function
* It is then evaluated if need be on the test data set, through evaluate function


```python
imageModel = Sequential()
```


```python
inputShape = (height,width,1)
imageModel.add(Convolution2D(16,5,5,border_mode="same",input_shape = inputShape))
imageModel.add(MaxPooling2D(pool_size=(2,2),border_mode="same"))
imageModel.add(Activation(activation="relu"))
imageModel.add(Convolution2D(36,5,5,border_mode="same"))
imageModel.add(MaxPooling2D(pool_size=(2,2),border_mode="same"))
imageModel.add(Activation(activation="relu"))
imageModel.add(Flatten())
imageModel.add(Dense(128))
imageModel.add(Dense(numClasses))
imageModel.add(Activation(activation="softmax"))
```


```python
imageModel.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
```

### Copying output of the fit command to a txt file
* this is done because keras was showing webSocketError after training for sometime.
    * and would then quit training midway.
    * this was an issue with the stdout of the command because of the progress bar.
* So if you get the same error, this is a workaround to that problem.
* ps: the training time also improved because of this workaround. 


```python
import sys
sys.stdout = open('keras_output.txt', 'w')
history = imageModel.fit(X_train, y_train, batch_size=128, nb_epoch=5, verbose=1,validation_split=0.33)
sys.stdout = sys.__stdout__
```

## model on test data set
* loss = 0.03
* accuracy = **<span style="color:green">98.97%</span>**


```python
imageModel.evaluate(X_test,y_test,verbose=0)
```




    [0.03110207096124068, 0.98970000000000002]




```python

```


```python

```
