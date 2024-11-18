"""
Using Keras, the API built on top of TensorFlow, we will build an image classifier using the fashion MNIST image data.
The dataset represents 70,000 greyscale images of 28x28 pixels each, with 10 classes.
"""
# Library imports
import tensorflow as tf  # Note that the Keras API is available through tf.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data using Keras
mnist_data = tf.keras.datasets.fashion_mnist.load_data()

# The data are already shuffled and split into a training set (60,000 images) and test set (10,000 images)
# We're going to hold out the last 5,000 images for validation
(feature_train_full, target_train_full), (feature_test, target_test) = mnist_data  # Build training and test sets
feature_train, target_train = feature_train_full[:-5000], target_train_full[:-5000]  # Remove rows used for validation
feature_validate, target_validate = feature_train_full[-5000:], target_train_full[-5000:]  # Create validation sets

'''
NOTE: When loading the fashion MNIST data with Keras instead of sklearn, a crucial difference is that the images are
represented as 28x28 arrays rather than as a one-dimensional array of size 784. Also, the pixel intensities are 
represented as integers from 0 to 255 rather than as floats.

Have a look at the shape and data type of the training set: 
'''
print(f'Training set shape: {feature_train.shape}')  # Prints (55000, 28, 28) - 55,000 28x28 arrays
print(f'Training set datatype: {feature_train.dtype}')  # Prints uint8 - Integer datatype

'''
For simplicity, we'll scale the pixel intensities down to a 0-1 range by dividing them by 255. 
Note that this will convert them to float-type.
'''
feature_train, feature_validate, feature_test = feature_train / 255., feature_validate / 255., feature_test / 255.

# We need a list of class names to use in our classification model:
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
    ]

# For example, the first image in the training set represents an ankle boot - this is its 'known label'
print(f'Label of first training set image: {class_names[target_train[0]]}')  # Prints 'Ankle boot'

# Now we'll build the neural net using Keras' sequential API. This will be a classifier with two hidden layers.
# Set random seed
tf.random.set_seed(4)

'''
Now, we instantiate a sequential neural net model. 
This is the simplest type of Keras neural net model - for neural nets which are composed of a single stack of layers.
This is what defines the 'sequential API'
'''
model = tf.keras.Sequential()

'''
Next, we build the first layer (the input layer) and add it to the model.
We specify the input shape, which doesn't include the batch size, only the shape of the instances.

Recall that the shape of each image is a 28x28 matrix.

Keras needs to know the shape of the inputs so it can determine the shape of the connection weight matrix of the
first hidden layer.
'''
model.add(tf.keras.layers.Input(shape=[28, 28]))

'''
Next, we add a 'flattening' layer. Its role is to convert each 28x28 representation of each image into a 1D array.
For example, if it receives a batch of shape [32, 28, 28], it will reshape it to [32, 784].
In other words, In other words, given input data X, it computes X.reshape(-1, 784).
This layer has no parameters - it only exists to do simple data preprocessing.
'''
model.add(tf.keras.layers.Flatten())

'''
Now we add a dense hidden layer containing 300 units/nodes. It uses the ReLU activation function.

Each 'Dense' layer manages its own weight matrix, which contains all of the connection weights between the nodes
and their inputs. It also manages a vector of bias terms (one per node).
'''
model.add(tf.keras.layers.Dense(300, activation='relu'))

'''
We add a second dense hidden layer with 100 nodes, also using the ReLU activation function. 
'''
model.add(tf.keras.layers.Dense(100, activation='relu'))

'''
Finally, we add the output layer with 10 nodes, one per class.
We can use the SoftMax activation function to compute class probabilities because this is a multi-class classification
problem, and in this instance the classes are exclusive.
'''
model.add(tf.keras.layers.Dense(10, activation='softmax'))

'''
OTHER THINGS TO NOTE:
- Specifying activation='relu' as a parameter is equivalent to writing activation = tf.keras.activations.relu as its
own line of code. Other activation functions are available in the tf.keras.activations package.

- Instead of adding layers one by one, as we did above, you might also choose to pass a list of layers when creating
the sequential model. If you do this, you can drop the explicit definition of the input layer and instead just specify
the input_shape parameter in your first layer. The following code is equivalent to what we did above:

model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
tf.keras.layers.Dense(300, activation='relu'),
tf.keras.layers.Dense(100, activation='relu')
tf.keras.layers.Dense(10, activation='softmax')
])
'''

'''
MODEL SUMMARY:
The model's .summary() method displays all of the model's layers, including each layer's name (which is automatically
generated unless you explicitly define it when creating the layer), its output shape ('None' means the batch size could
be anything), and its number of parameters.

The summary ends with the total number of parameters, including trainable and non-trainable parameters.
Note that in this implementation, we have only trainable parameters.
'''
print('Model summary:')
print(model.summary())

'''
LAYER PARAMETERS:
All of the parameters of a layer can be accessed using its get_weights() and set_weights() methods.
For a Dense layer, this includes the connection weights and the bias terms.

You'll note that the Dense layer initialized the connection weights randomly (required to break symmetry) and 
initialized the bias weights to zero.

If you want to use a different initialization method, you can explicitly set the kernel_initializer and bias_initializer
parameters when creating a layer.
'''
hidden_layer_1 = model.layers[1]  # Identify first hidden layer
print(f'Name of hidden layer 1: {hidden_layer_1.name}')  # Check that correct layer has been identified
weights, biases = hidden_layer_1.get_weights()
print(f'Weights: {weights}')
print(f'Shape of weights: {weights.shape}')
print(f'Biases: {biases}')
print(f'Shape of biases: {biases.shape}')

'''
IMPORTANT:
The shape of the weight matrix depends on the number of inputs, which is why we specified the input_shape when creating
the model. 

If you don't specify the input shape when creating the model, you're probably okay - Keras can wait until
it knows the input shape before it actually constructs the model parameters. This will happen either when you feed it
data (i.e. during training) or when you call its .build() method.

However, until the model parameters are built, you will not be able to either display the model parameters or save the 
model. So, if you know the input shape when creating the model, you should definitely specify it up front.
'''

'''
MODEL COMPILATION:
After the model is created, we need to call its .compile() method to specify the loss function and optimizer that we
want to use. Optionally, you can specify a list of extra metrics to compute during training and evaluation.

In this model, we will use a sparse categorical cross-entropy loss function because we have sparse labels - for each
instance, there is only a target class index (from 0 to 9, because there are ten classes and the classes are exclusive).

We will train the model using stochastic gradient descent as an optimizer. (Our backpropagation will be reverse-mode
autodiff plus stochastic gradient descent.)

Because this is a classifier, the most useful and comprehensible performance metric is accuracy, which as we know
represents the % of total classifications which were correct according to known-label comparisons. 
'''
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'''
MODEL TRAINING:
We are now ready to train the model. To do so, we simply call its .fit() method.

We pass this method the input features (feature_train) and the target classes (target_train). 

We also pass the number of epochs we want it to train for. Note: the default number of epochs is 1, which would 
certainly not be enough to reach model convergence.

We can also pass it a validation set. Keras will measure the loss and the extra metrics on the validation set at the
end of each epoch, which is useful in terms of seeing how well the model truly performs. If the performance on the
training set is much better than the performance on the validation set, we're overfitting to the training data.

The .fit() method returns a history object which contains the training parameters (history.params), the list of epochs
the model went through (history.epoch), and most importantly a dictionary (history.history) which contains the loss
and extra metrics it measured at the end of each epoch on the training and validation sets (if applicable).
'''
history = model.fit(feature_train, target_train, epochs=30, validation_data=(feature_validate, target_validate))

# You can convert the history dictionary to a Pandas dataframe and call its .plot() method to see your learning curves
pd.DataFrame(history.history).plot(figsize=(8, 8),
                                   xlim=[0, 30],
                                   ylim=[0, 1],
                                   grid=True,
                                   xlabel='Epoch',
                                   style=['r--', 'b-', 'b--', 'b-*'])
plt.show()

print('Model performance on test set:')
print(model.evaluate(feature_test, target_test))  # Evaluate model performance on test set

# Making predictions on new instances
feature_new = feature_test[:3]  # Select some images to use (not really 'new' per se but whatever)
target_probs = model.predict(feature_new)  # Generate probability predictions for those images
print(f'Predicted target probabilities: {target_probs.round(2)}')  # Print probabilities
target_preds = target_probs.argmax(axis=-1)  # Locate predicted class numbers
print(f'Predicted target classes: {target_preds}')  # Print predicted class numbers
print(f'Predicted target class names: {np.array(class_names)[target_preds]}')  # Print predicted class names
true_labels = target_test[:3]  # Fetch true labels
print(f'True labels: {true_labels}')  # Print true labels - the NN predicted all three examples correctly

