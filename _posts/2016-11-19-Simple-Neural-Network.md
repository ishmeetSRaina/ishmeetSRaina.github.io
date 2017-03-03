---
layout: post
title: Simple Neural Network
categories: [ComputerVision]
author: ISR
---

* This notebook shows how to create a simple neural network in tensorflow for handwritten digit classification on the classic MNIST database.
* The accuracy of model after around 5000 iterations on the data set is about 93%.
* The notebook carries explanation for tensorflow process.
<!--more-->

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import sys
%matplotlib inline
```

### Library info:



```python
print("Python = ",sys.version.split("|")[0])
print("Tensorflow = ", tf.__version__)
print("Numpy = ",np.version.version)
```

    Python =  3.5.2 
    Tensorflow =  0.12.0-rc0
    Numpy =  1.11.3


### Loading data


```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
```

    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting data/MNIST/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting data/MNIST/train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting data/MNIST/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting data/MNIST/t10k-labels-idx1-ubyte.gz



```python
print("length of training set :\t\t{}".format(len(data.train.labels)))
print("length of test set :\t\t{}".format(len(data.test.labels)))
print("length of validation set :\t\t{}".format(len(data.validation.labels)))
```

    length of training set :		55000
    length of test set :		10000
    length of validation set :		5000



```python
data.test.cls = np.array([label.argmax() for label in data.test.labels])
```


```python
data.test.cls[4:9]
```




    array([4, 1, 4, 9, 5])




```python
len(data.train.images[0])
```




    784




```python
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10
```


```python
# to plot images 
def plotDigits(digits, actualDigit, predDigit=None):
    
    # create a plot matrix for the digits
    noDigits = len(digits)
    row = int(math.sqrt(noDigits))
    col = math.ceil(noDigits/row)
    
    fig, axes = plt.subplots(row, col)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        if i<noDigits:
            # Plot digit image.
            ax.imshow(digits[i].reshape(img_shape), cmap='binary')
            
            # Show actual and predicted classes.
            if predDigit is None:
                xlabel = "Actual: {0}".format(actualDigit[i])
            else:
                xlabel = "Actual: {0}, Pred: {1}".format(actualDigit[i], predDigit[i])

            ax.set_xlabel(xlabel)
             
        ax.set_xticks([])
        ax.set_yticks([])
```

### plotting some test images


```python
# Get the first images from the test-set.
digits = data.test.images[0:15]

# Get the true classes for those images.
actualDigits = data.test.cls[0:15]

# Plot the images and labels using our helper-function above.
plotDigits(digits=digits, actualDigit=actualDigits)
```


![png](output_13_0.png)


# <span style="color:orange"> TensorFlow Graph</span>
The entire purpose of TensorFlow is to have a so-called computational graph that can be executed much more efficiently than if the same calculations were to be performed directly in Python. TensorFlow can be more efficient than NumPy because TensorFlow knows the entire computation graph that must be executed, while NumPy only knows the computation of a single mathematical operation at a time.  
TensorFlow can also **automatically calculate the gradients** that are needed to optimize the variables of the graph so as to make the model perform better.   
This is because the **graph is a combination of simple mathematical expressions** so the gradient of the entire graph can be calculated using the **chain-rule for derivatives**.

A TensorFlow graph consists of the following parts which will be detailed below:
* **Placeholder variables** used to change the input to the graph.
* **Model variables** that are going to be optimized so as to make the model perform better.
* **The model** which is essentially just a mathematical function that calculates some output given the input in the placeholder variables and the model variables.
* A **cost measure** that can be used to guide the optimization of the variables.
* An **optimization method** which updates the variables of the model.  



## <span style="color:green">1. Placeholder variables</span>

Placeholder variables = **input to the graph**
* **change each time we execute the graph**. 

In this we create 3 placeholder variables:

* placeholder variable for the <span style="color:blue">**input images**.</span>   
    * data-type = float32
    * shape is set to [None, img_size_flat]
        * None means that the tensor may hold an __arbitrary number of images__ 
        * each image being a vector of length img_size_flat.



```python
x = tf.placeholder(tf.float32,[None,img_size_flat])
```

* placeholder variable for the <span style="color:blue">**true labels** associated with the images</span>
    * shape = [None, num_classes] 


```python
y_true = tf.placeholder(tf.float32,[None,num_classes])
```

* placeholder variable for the <span style="color:blue">**true class** of each image</span>
    * dtype = int64 
    * shape = [None] 
        * means the placeholder variable is a one-dimensional vector of arbitrary length.


```python
y_true_cls = tf.placeholder(tf.int64,[None])
```

## <span style="color:green">2. Model Variables to be optimized</span>
* These are the variables that define the model 
* These would be changed when the optimization is done 
* We have 2 variables in this example
<div style="color:blue">
    1. Weights   
    2. Biases
</div>
* we use truncated_normal to randomize initial weights and biases


```python
weights = tf.Variable(tf.truncated_normal([img_size_flat,num_classes],stddev=0.001))
biases = tf.Variable(tf.truncated_normal([num_classes],stddev=0.001))
```


```python

```

## <span style="color:green">3. Model</span>
* for test purpose we use a simple mathematical model 
    * multiplies the images in the placeholder variable x with the weights and then adds the biases.  
* The result is a matrix of shape [num_images, num_classes] 
    * x has shape [num_images, img_size_flat] 
    * weights has shape [img_size_flat, num_classes], 
    * multiplication of those two matrices is a matrix with shape [num_images, num_classes] 
    * biases vector is added to each row of that matrix.  




```python
logits = tf.matmul(x,weights)+biases
```

* logits is a matrix
    * rows = num_images
    * cols = num_classes   
* logits[i,j] =  **likelihood** that the $i$'th input image is to be of the $j$'th class  

* we then normalize them using **softmax** function
    * each row of the logits matrix sums to one
    * each element is limited between [0,1]  
    
  
* store it in **y_pred**



```python
y_pred = tf.nn.softmax(logits)
```

* The predicted class  = column with the max probability  


* stored in **y_pred_cls**




```python
y_pred_cls = tf.arg_max(y_pred,dimension=1)
```

## <span style="color:green">4. Cost-function to be optimized</span>

comparing the predicted output of the model y_pred to the desired output y_true.  

* The cross-entropy is a performance measure used in classification. 
    * The cross-entropy is a continuous function that is always positive 
    * if the predicted output = actual output ==> cross-entropy equals zero.     
      
      
* The goal of optimization = **minimize cross-entropy** by changing the weights and biases



```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

```

* calculated the cross-entropy for each of the image classifications 
    * we have a measure of how well the model performs on **each image individually**
* But in order to use the cross-entropy to guide the optimization of the model's variables 
    * we need a single scalar value
    * so we simply take the average of the cross-entropy for all the image classifications.
      
* store it in **cost**


```python
cost = tf.reduce_mean(cross_entropy)
```

## <span style="color:green">5. Optimization method</span>

* we have a cost measure that must be minimized
    * create an optimizer

* In this example it is a simple Gradient Descent where the step-size is set to 0.5.  




```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
```

## Performance measures
### <span style="color:green">Accuracy  </span>




```python
correct_prediction = tf.equal(y_pred_cls, y_true_cls) # gives an array of True and False values

# convert False =0, True =1 
# calculate mean i.e. Total True/ total Values
# this gives accuracy

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
```


```python

```

# <span style="color:green">Tensorflow graph created</span>

# <span style="color:red"> Tensorflow run</span>

### Creating tensorflow session


```python
session = tf.Session()
```

### Initialize variables
The variables for weights and biases must be initialized before we start optimizing them.


```python
session.run(tf.global_variables_initializer())
```


```python
def checkWeights():
    w = session.run(weights)
    print(np.min(w),np.max(w))
```


```python
checkWeights()
```

    -0.00199919 0.00199818


### Helper-function to perform optimization iterations
taking mini batches of 100 instead of all 50,000 images


```python
batch_size = 100
```

Function for performing a number of optimization iterations so as to gradually improve the weights and biases of the model.  
In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples



```python
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        
    
```

### Helper-functions to show performance
Dict with the test-set data to be used as input to the TensorFlow graph.   
Note that we must use the correct names for the placeholder variables in the TensorFlow graph


```python
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
```

Function for printing the classification accuracy on the test-set.


```python
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
```

Printing confusion matrix


```python
def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
```

Function for plotting examples of images from the test-set that have been mis-classified.


```python
def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    # correct is a numpy array?
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plotDigits(digits=images[0:9],
                actualDigit = cls_true[0:9],
                predDigit =cls_pred[0:9])
```

#### Helper-function to plot the model weights
Function for plotting the weights of the model. 10 images are plotted, one for each digit that the model is trained to recognize.


```python
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    print(w_min, w_max)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
```

## Performance before any optimization



```python
print_accuracy()
```

    Accuracy on test-set: 9.4%



```python
plot_weights()
```

    -0.00199919 0.00199818



![png](output_63_1.png)



```python
print_confusion_matrix()
```

    [[ 28  52  23 861   0   3  10   1   0   2]
     [435  14  47 155   5   2   0 470   0   7]
     [314 197 139 260   6  64  32  16   1   3]
     [ 36 102  74 648  23  79   3  18   1  26]
     [  9  99  40 682   3  49   4  65   2  29]
     [ 88 122  34 479  14  71   2  72   1   9]
     [ 23 154 121 494  15  41  10  92   7   1]
     [305  15  61 410  19  56  11  14   1 136]
     [220  56 132 427  15  39   3  58   2  22]
     [ 70  18  17 774   1  40   1  75   2  11]]



![png](output_64_1.png)



```python
plot_example_errors()
```


![png](output_65_0.png)



```python

```

### running the optimizer once


```python
%timeit optimize(num_iterations=1)
print_accuracy()

```

    The slowest run took 201.90 times longer than the fastest. This could mean that an intermediate result is being cached.
    1 loop, best of 3: 1.77 ms per loop
    Accuracy on test-set: 50.4%



```python
plot_weights()
```

    -0.116125 0.0975963



![png](output_69_1.png)



```python
plot_example_errors()
```


![png](output_70_0.png)


### running optimizer for 10 instances
already ran for 1, so will run for 9 more


```python
%timeit optimize(num_iterations=9)
print_accuracy()
```

    100 loops, best of 3: 17.1 ms per loop
    Accuracy on test-set: 92.0%



```python
plot_weights()
```

    -0.967859 1.24459



![png](output_73_1.png)



```python
print_confusion_matrix()
```

    [[ 959    0    2    4    0    6    5    2    2    0]
     [   0 1106    2    3    1    4    3    2   14    0]
     [   5    9  897   32   12    5   11   11   44    6]
     [   2    0   11  944    0   18    0   10   18    7]
     [   1    2    3    3  935    0    7    2    7   22]
     [   9    2    2   53   12  773    9    3   24    5]
     [  11    3    4    2   16   24  892    3    3    0]
     [   1    8   19    9   10    1    0  944    3   33]
     [   4    8    5   33   11   39    8   10  850    6]
     [   9    8    1   14   47   10    0   15    7  898]]



![png](output_74_1.png)



```python
plot_example_errors()
```


![png](output_75_0.png)


### running optimizer for 100 reps


```python
%timeit optimize(num_iterations=90)
print_accuracy()
```

    10 loops, best of 3: 169 ms per loop
    Accuracy on test-set: 92.4%



```python
plot_weights()
```

    -1.15154 1.62311



![png](output_78_1.png)



```python
print_confusion_matrix()
```

    [[ 959    0    1    3    1    6    7    2    1    0]
     [   0 1110    3    2    0    1    4    2   13    0]
     [   7    9  909   19    9    3   13   10   46    7]
     [   2    1   13  926    1   19    3   10   27    8]
     [   1    2    3    1  917    0   11    4    6   37]
     [   9    1    2   42   12  753   14    6   42   11]
     [   9    3    4    2    9   11  915    2    3    0]
     [   1    9   19    7    6    1    0  937    4   44]
     [   4    9    6   18    8   18   12    6  880   13]
     [  10    8    1    8   19    5    0   14    8  936]]



![png](output_79_1.png)



```python
plot_example_errors()
```


![png](output_80_0.png)


### running optimizer for 1000 loops


```python
%timeit optimize(num_iterations=990)
print_accuracy()
```

    1 loop, best of 3: 1.74 s per loop
    Accuracy on test-set: 92.4%



```python
plot_weights()
```

    -1.33206 1.90809



![png](output_83_1.png)



```python
print_confusion_matrix()
```

    [[ 965    0    1    3    0    5    3    2    1    0]
     [   0 1119    2    1    0    2    4    2    5    0]
     [   7   13  915   18    6    4   15    8   40    6]
     [   4    1   15  928    0   17    4   11   18   12]
     [   1    3    7    2  916    0    8    3    5   37]
     [  10    3    2   43   10  761   17    9   28    9]
     [  10    3    5    2    9   12  914    2    1    0]
     [   1   10   20    8    8    1    0  946    2   32]
     [   9   13    7   30    9   26   12    9  846   13]
     [  10    8    1    9   27    5    0   18    6  925]]



![png](output_84_1.png)



```python
plot_example_errors()
```


![png](output_85_0.png)


### running optimizer for 5000 loops


```python
%timeit optimize(num_iterations=4000)
print_accuracy()
```

    1 loop, best of 3: 7.89 s per loop
    Accuracy on test-set: 92.6%



```python
plot_weights()
```

    -2.12269 2.37464



![png](output_88_1.png)



```python
print_confusion_matrix()
```

    [[ 958    0    0    4    1    7    6    3    1    0]
     [   0 1111    4    3    0    1    3    2   11    0]
     [   4    7  924   18   10    3   15    8   40    3]
     [   3    1   13  942    1   11    3   10   21    5]
     [   1    2    4    3  932    0    7    3    8   22]
     [   8    2    0   61   10  753   14    7   32    5]
     [   7    3    4    3    9   13  916    2    1    0]
     [   1    7   16   12    7    1    0  956    6   22]
     [   3   11    4   28    8   21   10    7  876    6]
     [   7    8    0   14   43    5    0   29   15  888]]



![png](output_89_1.png)



```python
plot_example_errors()
```


![png](output_90_0.png)



```python

```
