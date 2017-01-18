import tensorflow as tf
import numpy as np
import graph_util

SET_SIZE = 1000
MAX_NODE = 10
MIN_NODE = 6
VALIDATION_SET_SIZE = SET_SIZE // 10
NUM_HIDDEN = 10

BATCH_SIZE = 1000
EPOCH = 5000

def train_path():
    """ Trains an LSTM model to find the shortest path in a graph"""

    # FIRSTLY marshall all of the data for training
    dataset = graph_util.build_dataset(SET_SIZE, MIN_NODE, MAX_NODE)
    print("STATUS: Finished generating graph data.\n\tSet size: {0}\n\tMin node: {1}\n\tMax node: {2}".format(SET_SIZE, MIN_NODE, MAX_NODE))

    train_input = graph_util.gen_input_set(dataset)
    train_output = graph_util.gen_target_set(dataset)

    # Split the data into training and validation sets.
    val_input = train_input[:VALIDATION_SET_SIZE] # Everything up to
    train_input = train_input[VALIDATION_SET_SIZE:] # Everything after

    val_output = train_output[:VALIDATION_SET_SIZE]
    train_output = train_output[VALIDATION_SET_SIZE:]

    print("STATUS: Training and validation sets created.")

    # Create input and output placeholders of the form (batch size, sequence length, input
    # dimensions).The batch size will be kept as variable. The sequence length is unknown
    # as it depends on the graph that has been generated. The input is 22-D - ten
    # bits for each digit, two bits for the phase flag.
    data = tf.placeholder(tf.float32, [None, None, 22])
    # The output will be 20 bits
    target = tf.placeholder(tf.float32, [None, 20])

    # Create an LSTM cell with specified number of hidden units.
    cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, state_is_tuple=True)



train_path()