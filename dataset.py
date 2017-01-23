import constants
import numpy as np

class Dataset:
    """
    A class to hold all the dataset info together for easy access. Also provides
    functions for formatting the dataset for use in the model.
    """
    def __init__(self,
                 graph_list,
                 terminal_nodes,
                 paths,
                 lengths,
                 min_node,
                 max_node):

        self.graph_list = graph_list
        self.terminal_nodes = terminal_nodes
        self.paths = paths
        self.lengths = lengths
        self.min_node = min_node
        self.max_node = max_node
        self.max_path_length = self.max_node - 1
        self.max_edge_list_length = ((self.max_node) * (self.max_node - 1)) / 2
        self.max_input_length = self.max_edge_list_length + constants.QUERY_PHASE_LENGTH + constants.PLAN_PHASE_LENGTH + self.max_path_length


    def get_input_set(self):
        """
        Creates the input sequence for each graph based on the model used for learning.
        Of the form: description_phase + query_phase + plan_phase + answer_phase

        Returns:
            input_set: a list of encoded input sequences
        """

        # First encode the graphs into their one-hot equivalents with description phase flag 0,0
        encoded_graphs = []
        for graph in self.graph_list:
            edge_list = graph.edge_list
            encoded_graph = []
            for edge in edge_list:
                node_a = self.__decimal_to_onehot(edge[0])
                node_b = self.__decimal_to_onehot(edge[1])
                encoded_edge = [0, 0] + node_a + node_b
                encoded_graph.append(encoded_edge)
            encoded_graphs.append(encoded_graph)

        input_set = []
        # For each encoded graph, construct the full input sequence
        for i in range(0, len(encoded_graphs)):
            desc_phase = encoded_graphs[i]
            query_phase = [[0, 1] +
                           self.__decimal_to_onehot(self.terminal_nodes[i][0]) +
                           self.__decimal_to_onehot(self.terminal_nodes[i][1])]
            plan_phase = [[1, 0] + [0] * 20] * constants.PLAN_PHASE_LENGTH

            # This is really a stop-gap measure. The answer phase should feed the output from
            # the previous step as input to the network, but i dont know how to do that yet
            # so instead ill feed it nothing until I can get more things running.
            # This phase is 8 steps long becase the maximum number of nodes is 9,
            # therefore the longest shortest path is 8. Everything else will be zero-padded.
            # (I don't know if this will work as intended and it's a bit unclean,
            # but it's simpler than trying to figure out how to get the network to learn
            # to output a special "im done" flag of some sort)

            answer_phase = [[1, 1] + [0] * 20] * self.max_path_length


            # This represents the entire input sequence for a single graph and shortest path query
            input_sequence = desc_phase + query_phase + plan_phase + answer_phase

            # Pad the unused space with zeros
            padding = [[0] * constants.INPUT_SIZE] * int(self.max_input_length - len(input_sequence))
            input_sequence += padding
            input_set.append(np.array(input_sequence))

        return input_set

    def get_target_set(self):
        """
        Creates the target output for each graph based on the model used for learning.
        Because of how variable length sequences are being handled in my model,
        the answer phase is padded out to be the same size as the input sequence.
        The leading padding will be the length of the input sequence UP TO the answer
        phase. This is followed by the expected output of the answer phase, then further
        padding to make it the same length as the input.

        Returns:
            target: a list of encoded target sequences
        """
        target_set = []

        for i in range(len(self.graph_list)):
            path = self.paths[i]
            length = self.lengths[i]
            #print(path)
            #print(length)

            # create the leading padding
            leading_length = length + constants.QUERY_PHASE_LENGTH + constants.PLAN_PHASE_LENGTH
            leading_padding = [[0] * constants.OUTPUT_SIZE] * leading_length
            #print(len(leading_padding))
            # Actual answer phase
            encoded_path = []
            for edge in path:
                node_a = self.__decimal_to_onehot(edge[0])
                node_b = self.__decimal_to_onehot(edge[1])
                encoded_path.append(node_a + node_b)

            # Trailing padding
            trailing_length = self.max_input_length - len(leading_padding) - len(encoded_path)
            trailing_padding = [[0] * constants.OUTPUT_SIZE] * int(trailing_length)

            target_sequence = leading_padding + encoded_path + trailing_padding
            target_set.append(np.array(target_sequence))

        return target_set

    def __decimal_to_onehot(self, decimal_digit):
        one_hot = [0] * 10
        one_hot[-decimal_digit-1] = 1
        return one_hot


