import functools
import tensorflow as tf
import graph_util
from constants import *

def lazy_property(function):
    """
    Used to annotate functions that we only want computer once. Cached results
    are then used on subsequent calls.
    """
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class ShortestPathFinder:
    """
    The class containing the LSTM network that learns to solve the shortest path problem.
    """

    def __init__(self,
                 data,
                 target,
                 num_hidden=200,
                 num_layers=1,
                 learning_rate=0.001,
                 dropout_in=1.0,
                 dropout_out=1.0):

        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._dropout_in = dropout_in
        self._dropout_out = dropout_out
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        """
        This is used to get the actual length of the sequence during computation of the graph,
        dealing with the padding that is necessary for using the same placeholders for variable
        length sequences. It works by flattening the input vectos using the max function.
        This will be 0 for each input that is a vector of zeros (a pad). Then sign converts
        the max values to 1, keeping the zeros as zero. Now we can just sum the ones
        to get the length of the sequence.

        Returns:
            length: the length of the input sequences up to and including the answer phase
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        """
        The heavy lifting function. Runs the batch of data through the LSTM network and
        and generates the output for each time step. Then, flattens the output across all
        batches and does the regression for every timestep. It then splits each output into
        two and does a softmax over each set of ten logits to get an edge label. Sticks the
        two nodes back together and reshapes it to the form
        [batch_size x sequence_length x num_hidden]

        Returns:
            predictions: the predicitons for that batch of inputs
        """
        cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self._num_hidden,
                                                                     state_is_tuple=True),
                                             self._dropout_in,
                                             self._dropout_out)


        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers)

        output, _ = tf.nn.dynamic_rnn(
            cell,
            self.data,
            dtype=tf.float32,
            sequence_length=self.length
        )

        max_answer_length = int(self.target.get_shape()[1])
        output_size = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, output_size)

        # Flatten to apply same weights to all time steps.
        # -1 is used to infer the dimension
        output = tf.reshape(output, [-1, self._num_hidden])
        output_regr = tf.matmul(output, weight) + bias

        # Split the output into two shape [batch_size * sequence_length, 10] tensors,
        # one for each node of the edge label
        output_a, output_b = tf.split(1, 2, output_regr)
        prediction_a = tf.nn.softmax(output_a)
        prediction_b = tf.nn.softmax(output_b)

        # Rejoin the individual softmaxes into a single prediction
        prediction = tf.concat(1, [prediction_a, prediction_b])
        prediction = tf.reshape(prediction, [-1, max_answer_length, output_size])
        return prediction

    @lazy_property
    def loss(self):
        """
        Returns:
            The average cross-entropy loss accross the batch. It is the sum of the
            cross-entropy loss of each individual digit, summed accross the answer
            phase.
        """
        # Compute the cross entropy for each digit at each timestep
        digit_a, digit_b = tf.split(2, 2, self.prediction)
        target_a, target_b = tf.split(2, 2, self.target)

        # Get the cross entropy for each individual digit, summed.
        cross_entropy = (target_a * tf.log(digit_a)) + (target_b * tf.log(digit_b))

        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)

        # We don't care about the output leading up to OR after the answer. For this
        # reason we create a mask that looks at the target vector and for each output
        # that is all zeros (a pad) enters a 0 in the mask, and for any output that
        # contains something (an answer edge) enters a 1 in the mask.
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        # Now we multiply the cross entropy values by the mask to clear the irrelevant
        # entries.
        cross_entropy *= mask

        # Use the mask we just created to find the actual asnwer lengths.
        answer_length = tf.reduce_sum(mask, reduction_indices=1)

        # Average over actual answer lengths. Do this instead of using reduce_mean
        # because that would divied by the maximum input lengths (ie, the length of our input
        # and target placeholders).
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(answer_length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):
        """
        TODO compare optimizers, learning rates
        Returns
            Optimizer to be used in training
        """
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        return optimizer.minimize(self.loss)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        """
        Returns:
            weight and bias vectors used in regression.
        """
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def error(self):
        """
        Returns:
            The error rate of a batch. The error rate is defined as the
            percentage of the test set for which the network did not return a path
            exactly equal to the corresponding path in the target set.
        """
        pred_a, pred_b = tf.split(2, 2, self.prediction)
        target_a, target_b = tf.split(2, 2, self.target)

        # Is the same bit hot for each digit of input and output
        mistake_a = tf.not_equal(
            tf.argmax(target_a, 2), tf.argmax(pred_a, 2))

        mistake_b = tf.not_equal(
            tf.argmax(target_b, 2), tf.argmax(pred_b, 2))

        # Is either digit a mistake?
        mistake = tf.Tensor.__or__(mistake_a, mistake_b)
        # Cast to 1s and 0s
        mistake = tf.cast(mistake, tf.float32)
        # Zero out all the outputs that aren't in the target answer phase
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistake *= mask
        #answer_length = tf.reduce_sum(mask, reduction_indices=1)

        # Were any of the outputs wrong for the sequence?
        mistake = tf.reduce_max(mistake, reduction_indices=1)
        #mistake = mistake // answer_length
        return tf.reduce_mean(mistake)

def chunks(data, chunk_size):
    """
    Splits the data up into smaller chunks
    Args:
        data: A list of data to be split
        chunk_size: the size of the chunks the list is to be split into

    Returns:
        new_list: A list of lists each of size <= chunk_size
    """
    new_list = []
    for i in range(0, len(data), chunk_size):
        new_list.append(data[i: i + chunk_size])
    return new_list

def train_model(set_size,
                min_node,
                max_node,
                num_hidden,
                num_layers,
                learning_rate,
                num_epochs,
                batch_size,
                dropout_in,
                dropout_out):
    """
    Trains the network according to the input hyperparameters.

    Returns:
        error_list: A list containing the error rate for each epoch trained.
    """

    test_set_size = set_size // 10

    no_of_batches = int(set_size / batch_size)

    dset = graph_util.build_dataset(
        set_size,
        min_node,
        max_node
    )
    train_input = dset.get_input_set()
    train_output = dset.get_target_set()

    test_input = train_input[:test_set_size]
    test_output = train_output[:test_set_size]

    train_input = train_input[test_set_size:]
    train_output = train_output[test_set_size:]

    # Puny GT650m starts getting OOM exceptions around 10k test set size so
    # split it into small sections to calculate the error
    test_input_chunks = chunks(test_input, 1000)
    test_output_chunks = chunks(test_output, 1000)

    print("\nSTATUS: Finished generating graph data." +
          "\n\tSet size: {0}\n\tMin node: {1}\n\tMax node: {2}\n".format(set_size,
                                                                         min_node,
                                                                         max_node))

    # None is used for the batch size to be determined dynamically
    data = tf.placeholder(tf.float32, [None, dset.max_input_length, INPUT_SIZE])
    target = tf.placeholder(tf.float32, [None, dset.max_input_length, OUTPUT_SIZE])

    model = ShortestPathFinder(data,
                               target,
                               num_hidden,
                               num_layers,
                               learning_rate,
                               dropout_in,
                               dropout_out)
    print("\nSTATUS: Model initialized.\n")

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print("\nSTATUS: Session initialized, beginning training.\n")


    error_list = []
    for epoch in range(num_epochs):
        ptr = 0
        for _ in range(no_of_batches):
            inp = train_input[ptr : ptr + batch_size]
            out = train_output[ptr : ptr + batch_size]
            #print("INPUT: {0}".format(inp))
            #print("OUTPUT: {0}".format(out))
            sess.run(model.optimize, {data: inp, target: out})

        error = 0
        for i, _ in enumerate(test_input_chunks):
            error += sess.run(model.error,
                              {data: test_input_chunks[i],
                               target: test_output_chunks[i]})
        error /= len(test_input_chunks)
        error_list.append(error * 100)
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))

    # Close and reset everything
    sess.close()
    tf.reset_default_graph()
    print("\nSTATUS: Session closed, finished training.\n")
    return error_list
