"""
Based on the model implemented by Danijar Hafner at https://gist.github.com/danijar/d11c77c5565482e965d1919291044470
"""
import functools
import tensorflow as tf
import graph_util
import constants


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

    def __init__(self, data, target, num_hidden=20, num_layers=1):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        """
        This is used to get the actual length of the sequence during computation of the graph,
        dealing with the padding that is necessary for using the same placeholders.
        It works by flattening the input vectos using the max function. This will be 0 for
        each input that is a vector of zeros (a pad). Then sign converts the max values to 1,
        keeping the zeros as zero. Now we can just sum the ones to get the length of the sequence.
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
        two nodes back together and reshapes it to the form [batch_size x sequence_length x num_hidden]

        Returns:
            predictions: the predicitons for that batch of inputs
        """
        cell = tf.nn.rnn_cell.LSTMCell(self._num_hidden, state_is_tuple=True)
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
        # Compute the cross entropy for each timestep
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.loss)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

def train_model():
    num_epochs = 10
    batch_size = 10
    set_size = 100
    max_node = 10
    min_node = 6
    test_set_size = set_size // 10
    no_of_batches = int(set_size / batch_size)

    dset = graph_util.build_dataset(set_size, min_node, max_node)
    train_input = dset.get_input_set()
    train_output = dset.get_target_set()

    test_input = train_input[:test_set_size]
    test_output = train_output[:test_set_size]

    train_input = train_input[test_set_size:]
    train_output = train_output[test_set_size:]

    print("STATUS: Finished generating graph data." +
          "\n\tSet size: {0}\n\tMin node: {1}\n\tMax node: {2}".format(set_size,
                                                                       min_node,
                                                                       max_node))

    data = tf.placeholder(tf.float32, [None, dset.max_input_length, constants.INPUT_SIZE])
    target = tf.placeholder(tf.float32, [None, dset.max_input_length, constants.OUTPUT_SIZE])

    model = ShortestPathFinder(data, target)
    print("STATUS: Model initialized.")

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print("STATUS: Session initialized, beginning training.")

    for epoch in range(10):
        ptr = 0
        for j in range(no_of_batches):
            inp = train_input[ptr : ptr + batch_size]
            out = train_output[ptr : ptr + batch_size]

            sess.run(model.optimize, {data: inp, target: out})
        print("EPOCH {0}".format(epoch))

    print("STATUS: Finished training.")

if __name__ == '__main__':
    train_model()
