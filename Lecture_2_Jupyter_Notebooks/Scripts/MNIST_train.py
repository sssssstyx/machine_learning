
# coding: utf-8

# ## MNIST hand written digits exercise
# 
# https://en.wikipedia.org/wiki/MNIST_database 
# http://yann.lecun.com/exdb/mnist/
# 
# In this example a neural network is trained to recognize handwritten digits.
# 
# ### about tensorflow
# 
# Check this video for introduction to tensorflow concepts:
# 
# https://www.youtube.com/watch?v=JO0LwmIlWw0&index=2&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs
# 
# 
# ### Exercise 1: 
# 
# Use the prev_validation_loss parameter and break out of the training epoch if the validation loss is higher than in previous round. This way we can avoid overfitting. Read about overfitting from the Internet what it actually means.
# 
# ### Exercise 2: 
# 
# Tune your model's hyperparameters so that you will get validation accuracy of 98%. 
# 
#   * you can change the activation functions (now sigmoid is used).
#   * you can change the hidden layer width.
#   * you can add more hidden layers (depth).
#   * you can modify the learning rate.
#   * what else?
# 
# ###  Exercise 3
# 
# Calculate the test data accuracy. Test accuracy is usually calculated after the model is not changed anymore. The test accuracy is an estimate how well your model predicts for data that it has not seen before.
# 
# You can read the test input and target data like this:
# 
# input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
# 
# ### Exercise 4. 
# 
# Save the variables. Use tf.train.Saver(). Check detail from here: https://www.tensorflow.org/guide/saved_model
# 
# ### Exercise 5. 
# 
# Create a new Jupyter notebook. Copy the model without the training code to a the new notebook (you only need the output related parts (feedforward) to get the final output). Add code to load the saved variables. Test that loading works by calculating test data accuracy (same way as in exercise 3.
# 
# ### Exercise 6. 
# 
# Using paint or other drawing tool create a 28x28 image and draw a number there. Save the image and load it to the notebook created in exercise 5. Feed it to the model and check is the prediction correct.
# 
# Use following package to load the image:
# 
# import matplotlib.image as mimg
# import matplotlib.pyplot as plt
# 
# img = mimg.imread('image.png')
# 
# plt.imshow(img) <-- use this to visualize how the image looks like.
# 
# new_input = img[:,:,1] <--- we do not need RGB channels, just use one of the channels. 
# 
# new_input = new_input.reshape((1,28*28))  <-- you need to reshape to fit the input layer
# 

# ## Import relevant libraries

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# ## Data
# 
# TensorFLow includes a data provider for MNIST. This function automatically downloads the MNIST dataset to the chosen directory. The dataset is already split into training, validation, and test subsets. Furthermore, it preprocess it into a particularly simple and useful format. Every 28x28 image is flattened into a vector of length 28x28=784, where every value corresponds to the intensity of the color of the corresponding pixel. The samples are grayscale (but standardized from 0 to 1), so a value close to 0 is almost white and a value close to  is almost purely black. 
# 
# Since this is a classification problem, our targets are categorical. We must convert the categories using one-hot encoding. Each individual sample is converted to a vector of length 10 which has nine 0s and a single 1 at the position which corresponds to the correct answer. 
# 
# For instance, if the true answer is "3", the target will be [0,0,0,1,0,0,0,0,0,0] (counting from 0).
# 
# if the true answer is "9", the target will be [0,0,0,0,0,0,0,0,0,1] (counting from 0).
# 
# if the true answer is "0", the target will be [1,0,0,0,0,0,0,0,0,0] (counting from 0).
# 
# if the true answer is "7", the target will be [1,0,0,0,0,0,0,1,0,0] (counting from 0).
# 
# ...

# In[2]:


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# ## Outline for the model

# In[3]:


input_size = 784
hidden_layer_size = 500
output_size = 10

# Reset any variables left in memory from previous runs. This needs to be run before changing tf variables.
tf.reset_default_graph()

# Declare placeholders where the data will be fed into.
input_layer = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

# Weights and biases for the first linear combination between the inputs and the first hidden layer.
weight_1 = tf.get_variable('weight_1',[input_size, hidden_layer_size])
biases_1 = tf.get_variable('biases_1', [hidden_layer_size])

# Operation between the inputs and the first hidden layer.
# We've chosen Sigmoid as our activation function. You can try playing with different non-linearities. Relu will be better...
output_1 = tf.nn.relu(tf.matmul(input_layer, weight_1) + biases_1)

# Weights and biases for the second linear combination.
# This is between the first hiden layer and the output layer

# when adding new hidden layers add them here.
weight_2 = tf.get_variable('weight_2',[hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable('biases_2', [hidden_layer_size])
output_2 = tf.nn.relu(tf.matmul(output_1, weight_2) + biases_2)

# add 3 more hidden layers
weight_3 = tf.get_variable('weight_3',[hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable('biases_3', [hidden_layer_size])
output_3 = tf.nn.relu(tf.matmul(output_2, weight_3) + biases_3)

weight_4 = tf.get_variable('weight_4',[hidden_layer_size, hidden_layer_size])
biases_4 = tf.get_variable('biases_4', [hidden_layer_size])
output_4 = tf.nn.relu(tf.matmul(output_3, weight_4) + biases_4)

weight_5 = tf.get_variable('weight_5',[hidden_layer_size, output_size])
biases_5 = tf.get_variable('biases_5', [output_size])

# notice that we are not using an activation function for the last layer.
# the output is the preciction of our network. 
# We could use softmax activaction function here. Check next comments for reasons why not.
output = tf.matmul(output_4, weight_5) + biases_5


# ### Model optimization

# In[4]:


# Calculate the loss function for every output/target pair.
# The function used is the same as applying softmax to the last layer and then calculating cross entropy
# This function, however, combines them in a clever way, 
# which makes it both faster and more numerically stable (when dealing with very small numbers).
# Logits here means: unscaled probabilities (so, the outputs, before they are scaled by the softmax)
# Naturally, the labels are the targets.
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=output)

# Get the average loss
mean_loss = tf.reduce_mean(loss)

# Define the optimization step. 
# Using adaptive optimizers such as Adam in TensorFlow. This is the optimization algorithm.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)

# Get a 0 or 1 for every input in the batch indicating whether it output the correct answer out of the 10.
# Check the tf.argmax documentation how this works. You can also experiment by printing the values
# to visualize what is going on
out_equals_targets = tf.equal(tf.argmax(output,1), tf.argmax(targets,1))

# Accuracy of our model. Basically correct answers divided by the number of all answers.
accuracy = tf.reduce_mean(tf.cast(out_equals_targets,tf.float32))


# ### Model is ready
# 
# ## Train the model

# In[5]:


# Declare the session variable. In tensorflow all computations are done in a session. 
# In the jupyter notebook the InteractiveSession is easy to use...
sess = tf.InteractiveSession()


# In[6]:


# Initialize the variables. Default initializer is Xavier.
init = tf.global_variables_initializer()

# Exercise 4 solution.
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# init is run in session. Otherwise nothing happens...
sess.run(init)


# In[9]:


# Define the batch size (in what kind of chunks of data is feed to the model).
# In many cases we cannot all data at once. There might be too much data to fit in memory.
# For that reason we are dividing the data in batches.
batch_size = 150

# Calculate the number of batches per epoch for the training set.
# epoch is one round of training with all the training data.
batches_number = mnist.train._num_examples // batch_size

# How many epochs we want to train.
max_epocs = 50

# Keep track of the validation loss of the previous epoch.
# If the validation loss becomes increasing, we want to trigger early stopping.
# We initially set it at some arbitrarily high number to make sure we don't trigger it
# at the first epoch
prev_validation_loss = 999999.

# Create a loop for the epochs. Epoch_counter is a variable which automatically starts from 0.
for epoch_counter in range(max_epocs):
    
    # Keep track of the sum of batch losses in the epoch.
    curr_epoch_loss = 0.
    
    # Iterate over the batches in this epoch.
    for batch_counter in range (batches_number):
        # Input batch and target batch are assigned values from the train dataset, given a batch size
        input_batch, target_batch = mnist.train.next_batch(batch_size)        
        
        # Run the optimization step and get the mean loss for this batch.
        # Feed it with the inputs and the targets we just got from the train dataset
        _, batch_loss = sess.run([optimizer, mean_loss],
                                feed_dict={input_layer: input_batch, targets:target_batch})
        
        # Increment the sum of batch losses.
        curr_epoch_loss += batch_loss
        
    # So far curr_epoch_loss contained the sum of all batches inside the epoch
    # We want to find the average batch losses over the whole epoch
    # The average batch loss is a good proxy for the current epoch loss    
    curr_epoch_loss /= batches_number
    
    # At the end of each epoch, get the validation loss and accuracy
    # Get the input batch and the target batch from the validation dataset    
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
        
    validation_accuracy, validation_loss = sess.run([accuracy, mean_loss],
                                feed_dict={input_layer: input_batch, targets:target_batch})
    
    
    # Exercise 1 solution.
    if validation_loss > prev_validation_loss:
        break
    else:
        # Exercise 4 solution.
        # Save the variables to disk.
        saver.save(sess, "./tmp/model.ckpt")
        
    prev_validation_loss = curr_epoch_loss

    # print result each epoch... As an exercise, you can try to pretty print the float values. 
    # For example only 3 decimals is enough.
    print('epoc round: ' + str(epoch_counter)+
         ' training loss: ' + str(curr_epoch_loss) +
         ' validation loss: ' + str(validation_loss) +
         ' validation_accuracy: ' + str(validation_accuracy))
        
    
print('training ready')


# In[8]:


# Exercise 3 solution:
input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)

test_accuracy = sess.run(accuracy,
                         feed_dict={input_layer: input_batch, targets:target_batch})

print('test accuracy: ' + str(test_accuracy))

