from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
slim = tf.contrib.slim
from tensorflow.contrib.framework.python.ops.variables \
    import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

'''
The skipped items from Tensorflow to the slim version
1. placeholder
2. Session()
3. Optimize.minimize()


'''

# =========== Parameters ============
BETA = 0.002                                # L2 regularization constant
TRAINING_EPOCHS = 1000                      # EPOCHS
BATCH_SIZE = 100                            # Set to 1 for whole sample at once
DISPLAY_STEP = 100                          # Set to 1 for displaying all epoch
DROPOUT_RATE = 0.9
VARIANCE = 0.1                              # VARIANCE selection highly affects
# Learning rate information and configuration (Up to you to experiment)
INITIAL_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_OF_EPOCHS_BEFORE_DECAY = 1000
learn_r = 0.000001

# ============ Network Parameters ============
HIDDEN_LAYERS = [500, 400, 600, 500, 500]
N_CLASSES = 1


def load_data_set(name=None):
    """
    This function reads the data file and extracts the features and labelled
    values.
    Then according to that patient is dead, only those observations would be
    taken care
    for deep learning trainig.
  Args:
    Nothing
  Returns:
    `Numpy array`, extracted feature columns and label column.
  Example:
    >>> read_dataset()
    ( [[2.3, 2.4, 6.5],[2.3, 5.4,3.3]], [12, 82] )
    """
    data_feed = pd.read_csv('Brain_Integ_X.csv', skiprows=[0], header=None)
    labels_feed = pd.read_csv('Brain_Integ_Y.csv', skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    censored_survival = survival[censored == 1]
    censored_data = data[censored == 1]

    y = np.asarray(censored_survival)
    x = np.asarray(censored_data)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    return (x, y)


# ******************************************************************************


with tf.Graph().as_default() as graph:
    logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    data_x, data_y = load_data_set()
    data_x, data_y = shuffle(data_x, data_y, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                        test_size=0.20,
                                                        random_state=420)

    total_observations = x_train.shape[0]         # Total Samples
    input_features = x_train.shape[1]             # number of columns(features)

    ckpt_dir = './log/'
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)

    num_batches_per_epoch = total_observations / BATCH_SIZE
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(NUM_OF_EPOCHS_BEFORE_DECAY * num_steps_per_epoch)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create the model inference
    def multilayer_neural_network_model(inputs, HIDDEN_LAYERS, BETA,
                                        scope="deep_regression_model"):
        """Creates a deep regression model.

        Args:
            inputs: A node that yields a `Tensor` of size [total_observations,
            input_features].

        Returns:
            predictions: `Tensor` of shape [1] (scalar) of response.
            end_points: A dict of end points representing the hidden layers.
        """
        with tf.variable_scope(scope, 'deep_regression', [inputs]):
            end_points = {}
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(BETA)):
                net = slim.stack(inputs,
                                 slim.fully_connected,
                                 HIDDEN_LAYERS,
                                 scope='fc')
                end_points['fc'] = net
                predictions = slim.fully_connected(net, 1, activation_fn=None,
                                                   scope='prediction')
                end_points['out'] = predictions
                return predictions, end_points

    # Construct the grpah for forward pass
    pred, end_points = multilayer_neural_network_model(x_train, HIDDEN_LAYERS, BETA)

    # Print name and shape of each tensor.
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Print name and shape of parameter nodes  (values not yet initialized)
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Create the global step for monitoring the learning_rate and training.
    global_step = get_or_create_global_step()

    # Define your exponentially decaying learning rate
    lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    loss = tf.losses.mean_squared_error(tf.squeeze(pred), y_train)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()

    # optimizer = tf.train.AdamOptimizer(learning_rate=.001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_r)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=.001)

    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    final = slim.learning.train(
        train_tensor,
        ckpt_dir,
        save_summaries_secs=20,
        global_step=global_step
    )
