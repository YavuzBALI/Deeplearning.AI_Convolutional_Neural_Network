# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 1
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    input_shape = (1, 1, 2, 2)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    input_shape = (600, 64, 64, 3)
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tf.keras.layers.ZeroPadding2D(padding=(3,3), 
                         input_shape=(64, 64, 3), data_format="channels_last"),
            ## Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(32, kernel_size=(7,7), strides=(1, 1), padding='valid'),
            ## BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis=-1),
            ## ReLU
            tf.keras.layers.ReLU(),
            ## Max Pooling 2D with default parameters
            tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            ## Flatten layer
            tf.keras.layers.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_27 (ZeroPaddi (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_6 (ReLU)               (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 100ms/step - loss: 1.5351 - accuracy: 0.6617
    Epoch 2/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.2453 - accuracy: 0.8983
    Epoch 3/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1492 - accuracy: 0.9467
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1661 - accuracy: 0.9283
    Epoch 5/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1946 - accuracy: 0.9200
    Epoch 6/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1644 - accuracy: 0.9383
    Epoch 7/10
    38/38 [==============================] - 4s 92ms/step - loss: 0.0874 - accuracy: 0.9717
    Epoch 8/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0867 - accuracy: 0.9650
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1599 - accuracy: 0.9467
    Epoch 10/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0798 - accuracy: 0.9700





    <tensorflow.python.keras.callbacks.History at 0x7f3a52e69f10>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 29ms/step - loss: 0.2062 - accuracy: 0.9133





    [0.20616987347602844, 0.9133333563804626]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 10
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 2



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(8, kernel_size=(4,4), strides=(1, 1), padding='same')(input_img)
    ## RELU
    A1 =tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D( pool_size=(8, 8), strides=(8,8), padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(16, kernel_size=(2,2), strides=(1, 1), padding='same')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D( pool_size=(4,4), strides=(4,4), padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6, activation="softmax")(F)
    # YOUR CODE STARTS HERE
    
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_8 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_19 (ReLU)              (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_19 (MaxPooling (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_20 (ReLU)              (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_20 (MaxPooling (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_13 (Flatten)         (None, 64)                0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.8402 - accuracy: 0.1694 - val_loss: 1.7953 - val_accuracy: 0.1917
    Epoch 2/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7946 - accuracy: 0.1722 - val_loss: 1.7909 - val_accuracy: 0.1667
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7908 - accuracy: 0.1769 - val_loss: 1.7880 - val_accuracy: 0.1917
    Epoch 4/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7883 - accuracy: 0.1602 - val_loss: 1.7862 - val_accuracy: 0.1833
    Epoch 5/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7857 - accuracy: 0.1806 - val_loss: 1.7844 - val_accuracy: 0.2583
    Epoch 6/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7827 - accuracy: 0.2269 - val_loss: 1.7822 - val_accuracy: 0.2917
    Epoch 7/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7788 - accuracy: 0.2556 - val_loss: 1.7797 - val_accuracy: 0.3000
    Epoch 8/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7729 - accuracy: 0.2778 - val_loss: 1.7761 - val_accuracy: 0.3083
    Epoch 9/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7643 - accuracy: 0.2861 - val_loss: 1.7704 - val_accuracy: 0.3167
    Epoch 10/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7526 - accuracy: 0.3046 - val_loss: 1.7622 - val_accuracy: 0.2917
    Epoch 11/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7362 - accuracy: 0.3370 - val_loss: 1.7520 - val_accuracy: 0.3000
    Epoch 12/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7139 - accuracy: 0.3667 - val_loss: 1.7368 - val_accuracy: 0.2833
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6854 - accuracy: 0.3824 - val_loss: 1.7177 - val_accuracy: 0.3000
    Epoch 14/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6464 - accuracy: 0.3815 - val_loss: 1.6919 - val_accuracy: 0.3333
    Epoch 15/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.6031 - accuracy: 0.3852 - val_loss: 1.6644 - val_accuracy: 0.3583
    Epoch 16/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.5561 - accuracy: 0.4093 - val_loss: 1.6343 - val_accuracy: 0.3833
    Epoch 17/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.5073 - accuracy: 0.4296 - val_loss: 1.5982 - val_accuracy: 0.4250
    Epoch 18/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.4586 - accuracy: 0.4463 - val_loss: 1.5614 - val_accuracy: 0.4250
    Epoch 19/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.4131 - accuracy: 0.4694 - val_loss: 1.5210 - val_accuracy: 0.4333
    Epoch 20/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.3706 - accuracy: 0.4917 - val_loss: 1.4833 - val_accuracy: 0.4417
    Epoch 21/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.3318 - accuracy: 0.5074 - val_loss: 1.4467 - val_accuracy: 0.4417
    Epoch 22/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.2961 - accuracy: 0.5278 - val_loss: 1.4128 - val_accuracy: 0.4750
    Epoch 23/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.2626 - accuracy: 0.5352 - val_loss: 1.3815 - val_accuracy: 0.4500
    Epoch 24/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2308 - accuracy: 0.5454 - val_loss: 1.3493 - val_accuracy: 0.4583
    Epoch 25/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.2003 - accuracy: 0.5667 - val_loss: 1.3184 - val_accuracy: 0.4750
    Epoch 26/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1709 - accuracy: 0.5843 - val_loss: 1.2881 - val_accuracy: 0.4833
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1440 - accuracy: 0.5944 - val_loss: 1.2609 - val_accuracy: 0.4917
    Epoch 28/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.1178 - accuracy: 0.6037 - val_loss: 1.2328 - val_accuracy: 0.5000
    Epoch 29/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0925 - accuracy: 0.6139 - val_loss: 1.2072 - val_accuracy: 0.5083
    Epoch 30/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0687 - accuracy: 0.6213 - val_loss: 1.1817 - val_accuracy: 0.5167
    Epoch 31/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0451 - accuracy: 0.6370 - val_loss: 1.1579 - val_accuracy: 0.5250
    Epoch 32/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0230 - accuracy: 0.6463 - val_loss: 1.1342 - val_accuracy: 0.5250
    Epoch 33/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0012 - accuracy: 0.6611 - val_loss: 1.1086 - val_accuracy: 0.5750
    Epoch 34/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9802 - accuracy: 0.6676 - val_loss: 1.0864 - val_accuracy: 0.5750
    Epoch 35/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9594 - accuracy: 0.6741 - val_loss: 1.0646 - val_accuracy: 0.6000
    Epoch 36/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.9394 - accuracy: 0.6722 - val_loss: 1.0462 - val_accuracy: 0.6167
    Epoch 37/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.9193 - accuracy: 0.6824 - val_loss: 1.0271 - val_accuracy: 0.6417
    Epoch 38/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8991 - accuracy: 0.6935 - val_loss: 1.0094 - val_accuracy: 0.6500
    Epoch 39/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8799 - accuracy: 0.7102 - val_loss: 0.9902 - val_accuracy: 0.6417
    Epoch 40/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.8622 - accuracy: 0.7185 - val_loss: 0.9741 - val_accuracy: 0.6417
    Epoch 41/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.8457 - accuracy: 0.7352 - val_loss: 0.9576 - val_accuracy: 0.6500
    Epoch 42/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8292 - accuracy: 0.7389 - val_loss: 0.9439 - val_accuracy: 0.6667
    Epoch 43/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8133 - accuracy: 0.7454 - val_loss: 0.9309 - val_accuracy: 0.6833
    Epoch 44/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.7979 - accuracy: 0.7472 - val_loss: 0.9184 - val_accuracy: 0.6833
    Epoch 45/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.7830 - accuracy: 0.7528 - val_loss: 0.9066 - val_accuracy: 0.6833
    Epoch 46/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7688 - accuracy: 0.7593 - val_loss: 0.8959 - val_accuracy: 0.6917
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7553 - accuracy: 0.7630 - val_loss: 0.8845 - val_accuracy: 0.7000
    Epoch 48/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.7421 - accuracy: 0.7704 - val_loss: 0.8746 - val_accuracy: 0.7000
    Epoch 49/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7290 - accuracy: 0.7778 - val_loss: 0.8627 - val_accuracy: 0.7000
    Epoch 50/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7158 - accuracy: 0.7861 - val_loss: 0.8526 - val_accuracy: 0.6917
    Epoch 51/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7032 - accuracy: 0.7917 - val_loss: 0.8417 - val_accuracy: 0.7000
    Epoch 52/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6912 - accuracy: 0.8019 - val_loss: 0.8337 - val_accuracy: 0.7083
    Epoch 53/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6798 - accuracy: 0.8028 - val_loss: 0.8237 - val_accuracy: 0.7167
    Epoch 54/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6683 - accuracy: 0.8019 - val_loss: 0.8162 - val_accuracy: 0.7250
    Epoch 55/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6577 - accuracy: 0.8028 - val_loss: 0.8072 - val_accuracy: 0.7250
    Epoch 56/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6471 - accuracy: 0.8065 - val_loss: 0.7988 - val_accuracy: 0.7250
    Epoch 57/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6367 - accuracy: 0.8074 - val_loss: 0.7904 - val_accuracy: 0.7417
    Epoch 58/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6269 - accuracy: 0.8083 - val_loss: 0.7822 - val_accuracy: 0.7417
    Epoch 59/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6173 - accuracy: 0.8093 - val_loss: 0.7737 - val_accuracy: 0.7500
    Epoch 60/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6077 - accuracy: 0.8176 - val_loss: 0.7655 - val_accuracy: 0.7500
    Epoch 61/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5979 - accuracy: 0.8185 - val_loss: 0.7569 - val_accuracy: 0.7583
    Epoch 62/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5879 - accuracy: 0.8222 - val_loss: 0.7472 - val_accuracy: 0.7583
    Epoch 63/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5787 - accuracy: 0.8259 - val_loss: 0.7403 - val_accuracy: 0.7583
    Epoch 64/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5703 - accuracy: 0.8278 - val_loss: 0.7327 - val_accuracy: 0.7583
    Epoch 65/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5625 - accuracy: 0.8296 - val_loss: 0.7266 - val_accuracy: 0.7583
    Epoch 66/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5545 - accuracy: 0.8343 - val_loss: 0.7192 - val_accuracy: 0.7583
    Epoch 67/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5472 - accuracy: 0.8352 - val_loss: 0.7123 - val_accuracy: 0.7583
    Epoch 68/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5398 - accuracy: 0.8361 - val_loss: 0.7057 - val_accuracy: 0.7667
    Epoch 69/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5325 - accuracy: 0.8361 - val_loss: 0.6990 - val_accuracy: 0.7667
    Epoch 70/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5256 - accuracy: 0.8426 - val_loss: 0.6937 - val_accuracy: 0.7667
    Epoch 71/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5190 - accuracy: 0.8463 - val_loss: 0.6882 - val_accuracy: 0.7667
    Epoch 72/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5122 - accuracy: 0.8481 - val_loss: 0.6826 - val_accuracy: 0.7667
    Epoch 73/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5061 - accuracy: 0.8491 - val_loss: 0.6770 - val_accuracy: 0.7667
    Epoch 74/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4999 - accuracy: 0.8509 - val_loss: 0.6713 - val_accuracy: 0.7750
    Epoch 75/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4941 - accuracy: 0.8519 - val_loss: 0.6664 - val_accuracy: 0.7750
    Epoch 76/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4881 - accuracy: 0.8546 - val_loss: 0.6611 - val_accuracy: 0.7750
    Epoch 77/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4825 - accuracy: 0.8556 - val_loss: 0.6575 - val_accuracy: 0.7750
    Epoch 78/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4775 - accuracy: 0.8546 - val_loss: 0.6530 - val_accuracy: 0.7750
    Epoch 79/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4722 - accuracy: 0.8556 - val_loss: 0.6487 - val_accuracy: 0.7750
    Epoch 80/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4673 - accuracy: 0.8556 - val_loss: 0.6445 - val_accuracy: 0.7833
    Epoch 81/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4622 - accuracy: 0.8556 - val_loss: 0.6398 - val_accuracy: 0.7833
    Epoch 82/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4573 - accuracy: 0.8593 - val_loss: 0.6352 - val_accuracy: 0.7917
    Epoch 83/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4524 - accuracy: 0.8611 - val_loss: 0.6315 - val_accuracy: 0.8000
    Epoch 84/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4476 - accuracy: 0.8648 - val_loss: 0.6263 - val_accuracy: 0.8000
    Epoch 85/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4431 - accuracy: 0.8667 - val_loss: 0.6221 - val_accuracy: 0.8000
    Epoch 86/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4390 - accuracy: 0.8685 - val_loss: 0.6187 - val_accuracy: 0.8000
    Epoch 87/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4347 - accuracy: 0.8694 - val_loss: 0.6145 - val_accuracy: 0.8083
    Epoch 88/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4307 - accuracy: 0.8704 - val_loss: 0.6100 - val_accuracy: 0.8083
    Epoch 89/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4267 - accuracy: 0.8731 - val_loss: 0.6029 - val_accuracy: 0.8083
    Epoch 90/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4224 - accuracy: 0.8769 - val_loss: 0.6006 - val_accuracy: 0.8083
    Epoch 91/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4193 - accuracy: 0.8769 - val_loss: 0.5934 - val_accuracy: 0.8083
    Epoch 92/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4152 - accuracy: 0.8796 - val_loss: 0.5931 - val_accuracy: 0.8083
    Epoch 93/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4117 - accuracy: 0.8824 - val_loss: 0.5837 - val_accuracy: 0.8083
    Epoch 94/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4069 - accuracy: 0.8852 - val_loss: 0.5830 - val_accuracy: 0.8167
    Epoch 95/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4042 - accuracy: 0.8833 - val_loss: 0.5758 - val_accuracy: 0.8167
    Epoch 96/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4001 - accuracy: 0.8861 - val_loss: 0.5748 - val_accuracy: 0.8250
    Epoch 97/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.3977 - accuracy: 0.8870 - val_loss: 0.5678 - val_accuracy: 0.8250
    Epoch 98/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3936 - accuracy: 0.8870 - val_loss: 0.5662 - val_accuracy: 0.8250
    Epoch 99/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.3912 - accuracy: 0.8889 - val_loss: 0.5570 - val_accuracy: 0.8250
    Epoch 100/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.3864 - accuracy: 0.8917 - val_loss: 0.5581 - val_accuracy: 0.8250


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.840185284614563,
      1.794647455215454,
      1.7907906770706177,
      1.7883182764053345,
      1.7857441902160645,
      1.7827032804489136,
      1.778751015663147,
      1.7729169130325317,
      1.7642576694488525,
      1.752610683441162,
      1.7362014055252075,
      1.7139451503753662,
      1.685383915901184,
      1.6464444398880005,
      1.6030582189559937,
      1.556130051612854,
      1.5073190927505493,
      1.4586269855499268,
      1.4131404161453247,
      1.37056303024292,
      1.3318079710006714,
      1.2960838079452515,
      1.2625913619995117,
      1.2307888269424438,
      1.2003403902053833,
      1.1709171533584595,
      1.1439926624298096,
      1.1177544593811035,
      1.0925151109695435,
      1.0686917304992676,
      1.0451024770736694,
      1.0229699611663818,
      1.0012426376342773,
      0.980248212814331,
      0.9593843817710876,
      0.9394087195396423,
      0.9193252921104431,
      0.8990949392318726,
      0.8799479603767395,
      0.8622098565101624,
      0.8456898927688599,
      0.8291506171226501,
      0.8132513165473938,
      0.7978788018226624,
      0.7830411791801453,
      0.7688257098197937,
      0.7552793025970459,
      0.742115318775177,
      0.7289931178092957,
      0.7157979011535645,
      0.7031542062759399,
      0.6911651492118835,
      0.679776132106781,
      0.6683364510536194,
      0.6576530933380127,
      0.6470881700515747,
      0.6366515755653381,
      0.6269297003746033,
      0.6172630190849304,
      0.607677161693573,
      0.597926139831543,
      0.5878796577453613,
      0.5787456035614014,
      0.5703180432319641,
      0.5624678134918213,
      0.554520308971405,
      0.5471718907356262,
      0.539798378944397,
      0.5324582457542419,
      0.5255520939826965,
      0.5189767479896545,
      0.5122269988059998,
      0.5061160326004028,
      0.4999338686466217,
      0.4940938949584961,
      0.4881127178668976,
      0.4825161397457123,
      0.47749683260917664,
      0.4722244143486023,
      0.4672953188419342,
      0.4621744155883789,
      0.4573086202144623,
      0.4524274170398712,
      0.447635680437088,
      0.4431319236755371,
      0.4390339255332947,
      0.43470022082328796,
      0.4307497441768646,
      0.42669248580932617,
      0.4223967492580414,
      0.4193088710308075,
      0.4151880145072937,
      0.4116976857185364,
      0.4069259762763977,
      0.40419670939445496,
      0.4000746011734009,
      0.39773455262184143,
      0.3935672640800476,
      0.39122965931892395,
      0.38643208146095276],
     'accuracy': [0.16944444179534912,
      0.17222222685813904,
      0.17685185372829437,
      0.16018518805503845,
      0.1805555522441864,
      0.22685185074806213,
      0.25555557012557983,
      0.2777777910232544,
      0.28611111640930176,
      0.3046296238899231,
      0.33703702688217163,
      0.36666667461395264,
      0.38240739703178406,
      0.38148146867752075,
      0.385185182094574,
      0.40925925970077515,
      0.4296296238899231,
      0.4462963044643402,
      0.4694444537162781,
      0.49166667461395264,
      0.5074074268341064,
      0.5277777910232544,
      0.5351851582527161,
      0.5453703999519348,
      0.5666666626930237,
      0.5842592716217041,
      0.5944444537162781,
      0.6037036776542664,
      0.6138888597488403,
      0.6212962865829468,
      0.6370370388031006,
      0.6462963223457336,
      0.6611111164093018,
      0.6675925850868225,
      0.6740740537643433,
      0.6722221970558167,
      0.6824073791503906,
      0.6935185194015503,
      0.710185170173645,
      0.7185184955596924,
      0.7351852059364319,
      0.7388888597488403,
      0.7453703880310059,
      0.7472222447395325,
      0.7527777552604675,
      0.7592592835426331,
      0.7629629373550415,
      0.770370364189148,
      0.7777777910232544,
      0.7861111164093018,
      0.7916666865348816,
      0.8018518686294556,
      0.8027777671813965,
      0.8018518686294556,
      0.8027777671813965,
      0.8064814805984497,
      0.8074073791503906,
      0.8083333373069763,
      0.8092592358589172,
      0.8175926208496094,
      0.8185185194015503,
      0.8222222328186035,
      0.8259259462356567,
      0.8277778029441833,
      0.8296296000480652,
      0.8342592716217041,
      0.835185170173645,
      0.8361111283302307,
      0.8361111283302307,
      0.8425925970077515,
      0.8462963104248047,
      0.8481481671333313,
      0.8490740656852722,
      0.8509259223937988,
      0.8518518805503845,
      0.854629635810852,
      0.855555534362793,
      0.854629635810852,
      0.855555534362793,
      0.855555534362793,
      0.855555534362793,
      0.8592592477798462,
      0.8611111044883728,
      0.864814817905426,
      0.8666666746139526,
      0.8685185313224792,
      0.8694444298744202,
      0.8703703880310059,
      0.8731481432914734,
      0.8768518567085266,
      0.8768518567085266,
      0.8796296119689941,
      0.8824074268341064,
      0.885185182094574,
      0.8833333253860474,
      0.8861111402511597,
      0.8870370388031006,
      0.8870370388031006,
      0.8888888955116272,
      0.8916666507720947],
     'val_loss': [1.7953031063079834,
      1.7909374237060547,
      1.7879809141159058,
      1.7861895561218262,
      1.7843849658966064,
      1.7822197675704956,
      1.7797240018844604,
      1.7760777473449707,
      1.7704029083251953,
      1.7621636390686035,
      1.751979947090149,
      1.7367538213729858,
      1.717675805091858,
      1.6919070482254028,
      1.6643849611282349,
      1.6343246698379517,
      1.5982451438903809,
      1.5614235401153564,
      1.5209904909133911,
      1.4832653999328613,
      1.4466928243637085,
      1.412841558456421,
      1.3814595937728882,
      1.349303960800171,
      1.318414568901062,
      1.2880847454071045,
      1.2609171867370605,
      1.2327930927276611,
      1.2071959972381592,
      1.1817346811294556,
      1.1579077243804932,
      1.1342288255691528,
      1.1085768938064575,
      1.0863666534423828,
      1.064591884613037,
      1.0462054014205933,
      1.0271283388137817,
      1.0093533992767334,
      0.9902185797691345,
      0.9740803837776184,
      0.957560122013092,
      0.9439145922660828,
      0.9309371709823608,
      0.9184374213218689,
      0.9066095948219299,
      0.895875096321106,
      0.8844727873802185,
      0.8745564818382263,
      0.8626759052276611,
      0.8525533676147461,
      0.8417255878448486,
      0.8336856365203857,
      0.8236517310142517,
      0.8161551356315613,
      0.8071843385696411,
      0.7987669706344604,
      0.7903597354888916,
      0.7822101712226868,
      0.7736664414405823,
      0.7655405402183533,
      0.7568837404251099,
      0.7472072839736938,
      0.7403410077095032,
      0.7327304482460022,
      0.7265978455543518,
      0.7192411422729492,
      0.7122744917869568,
      0.7057233452796936,
      0.6989889740943909,
      0.6937354803085327,
      0.6881514191627502,
      0.6825825572013855,
      0.6770104765892029,
      0.6712847352027893,
      0.6664199829101562,
      0.6610947847366333,
      0.657485842704773,
      0.6530435681343079,
      0.6487117409706116,
      0.6445069909095764,
      0.6398212313652039,
      0.6351934671401978,
      0.6315317153930664,
      0.6262741088867188,
      0.622062087059021,
      0.6187370419502258,
      0.6145433783531189,
      0.6100454926490784,
      0.6029231548309326,
      0.6005542874336243,
      0.5933768153190613,
      0.5930893421173096,
      0.5836824774742126,
      0.5829529166221619,
      0.5758167505264282,
      0.5748254060745239,
      0.5677868127822876,
      0.5662129521369934,
      0.5569708347320557,
      0.5581278204917908],
     'val_accuracy': [0.19166666269302368,
      0.1666666716337204,
      0.19166666269302368,
      0.18333333730697632,
      0.25833332538604736,
      0.2916666567325592,
      0.30000001192092896,
      0.3083333373069763,
      0.3166666626930237,
      0.2916666567325592,
      0.30000001192092896,
      0.28333333134651184,
      0.30000001192092896,
      0.3333333432674408,
      0.3583333194255829,
      0.38333332538604736,
      0.42500001192092896,
      0.42500001192092896,
      0.4333333373069763,
      0.4416666626930237,
      0.4416666626930237,
      0.4749999940395355,
      0.44999998807907104,
      0.4583333432674408,
      0.4749999940395355,
      0.4833333194255829,
      0.49166667461395264,
      0.5,
      0.5083333253860474,
      0.5166666507720947,
      0.5249999761581421,
      0.5249999761581421,
      0.574999988079071,
      0.574999988079071,
      0.6000000238418579,
      0.6166666746139526,
      0.6416666507720947,
      0.6499999761581421,
      0.6416666507720947,
      0.6416666507720947,
      0.6499999761581421,
      0.6666666865348816,
      0.6833333373069763,
      0.6833333373069763,
      0.6833333373069763,
      0.6916666626930237,
      0.699999988079071,
      0.699999988079071,
      0.699999988079071,
      0.6916666626930237,
      0.699999988079071,
      0.7083333134651184,
      0.7166666388511658,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7416666746139526,
      0.7416666746139526,
      0.75,
      0.75,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7666666507720947,
      0.7666666507720947,
      0.7666666507720947,
      0.7666666507720947,
      0.7666666507720947,
      0.7666666507720947,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7833333611488342,
      0.7833333611488342,
      0.7916666865348816,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8166666626930237,
      0.8166666626930237,
      0.824999988079071,
      0.824999988079071,
      0.824999988079071,
      0.824999988079071,
      0.824999988079071]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_41_1.png)



![png](output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
