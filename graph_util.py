'''
Graph utility class for generating random graph data, finding shortest paths, and formatting
the data for input to the network
'''
import random
import graph as gr
import math
import numpy as np


def gen_graph_data(set_size, min_node, max_node):
    """
    Generates graph data for dataset. Graph sizes are bounded above and below by max_node
    and min_node respectively (inclusive). Overall set size is controlled by set_size
    """
    graph_data = []
    for _ in range(0, set_size):
        graph_size = random.randint(min_node, max_node)
        graph_data.append(gr.Graph(graph_size))

    return graph_data


def dijkstra(graph, start):
    """
    Implementation of Dijkstra's algorithm that finds the distances
    of the shortest paths between a start node and every other node in the graph
    """
    nodes = graph.get_nodes()
    adj = graph.get_adjacency()
    Q = []

    dist = [0] * len(nodes)
    prev = [0] * len(nodes)

    for node in nodes:
        dist[node] = math.inf
        prev[node] = None
        Q.append(node)

    dist[start] = 0

    while len(Q) != 0:
        min_d = math.inf
        u = 0
        for node in Q:
            if dist[node] < min_d:
                u = node
        Q.remove(u)
        neighbours = adj[u]
        for v in range(0, len(neighbours)):
            if v in Q and neighbours[v] == 1:
                alt = dist[u] + 1
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    return dist, prev

def shortest_path(graph, start, end):
    """
    Finds the shortest path between given start and end nodes of a graph, using dijkstra.
    """
    _, prev = dijkstra(graph, start)
    path = []
    path.append(end)
    while end != start:
        end = prev[end]
        path.append(end)

    path.reverse()
    path_as_edge_list = []
    for i in range(0, len(path)-1):
        path_as_edge_list.append([path[i], path[i + 1]])

    return path_as_edge_list

def gen_shortest_paths(graph_data):
    """
    Goes through every graph in the dataset, randomly chooses two nodes and finds the
    shortest path between them. Returns the list of terminal pairs and a list of corresponding
    shortest paths.
    """
    terminal_pairs = []
    shortest_paths = []
    for i in range(0, len(graph_data)):
        graph = graph_data[i]
        terminal_pairs.append(random.sample(graph.get_nodes(), 2))
        shortest_paths.append(shortest_path(graph, terminal_pairs[i][0], terminal_pairs[i][1]))

    return terminal_pairs, shortest_paths

def build_dataset(set_size, min_node, max_node):
    """
    Builds the entire dataset. Returns a single list with three
    dimensions: graph data, terminal nodes, shortest path between terminal nodes.
    """
    dataset = []
    dataset.append(gen_graph_data(set_size, min_node, max_node))
    terminal_nodes, shortest_paths = gen_shortest_paths(dataset[0])
    dataset.append(terminal_nodes)
    dataset.append(shortest_paths)

    return dataset

def decimal_to_onehot(decimal_digit):
    """
    Converts a decimal digit (0-9) to its one-hot equivalent (e.g. 1 is 0000000010)
    """
    one_hot = [0] * 10
    one_hot[-decimal_digit-1] = 1
    return one_hot

def __encode_graph_data(graphs):
    """
    Accepts a list of graphs and returns the same list of graphs encoded with
    one-hot encodings. An optional phase_flag is prepended in the case of input
    graph data (it will not be needed for generating the target set).
    """
    encoded_graphs = []
    for graph in graphs:
        edge_list = graph.get_graph()
        encoded_graph = []
        for edge in edge_list:
            encoded_edge = [0, 0] + decimal_to_onehot(edge[0]) + decimal_to_onehot(edge[1])
            encoded_graph.append(encoded_edge)
        encoded_graphs.append(encoded_graph)

    return encoded_graphs

def gen_input_set(dataset):
    """
    Creates the input sequence for each graph based on the model used for learning.
    Of the form: description_phase + query_phase + plan_phase + answer_phase
    """
    encoded_graphs = __encode_graph_data(dataset[0])
    input_set = []
    for i in range(0, len(encoded_graphs)):
        desc_phase = encoded_graphs[i]
        query_phase = [[0,1] + decimal_to_onehot(dataset[1][i][0]) + decimal_to_onehot(dataset[1][i][1])]
        plan_phase = [[1, 0] + [0] * 20] * 10 # ten timesteps with no input

        # This is really a stop-gap measure. The answer phase should feed the output from
        # the previous step as input to the network, but i dont know how to do that yet
        # so instead ill feed it nothing until I can get more things running.
        # This phase is 8 steps long becase the maximum number of nodes is 9,
        # therefore the longest shortest path is 8. Everything else will be zero-padded.
        # (I don't know if this will work as intended and it's a bit unclean,
        # but it's simpler than trying to figure out how to get the network to learn
        # to output a special "im done" flag of some sort)
        answer_phase = [[1, 1] + [0] * 20] * 9

        # This represents the entire input sequence for a single graph and shortest path query
        input_sequence = desc_phase + query_phase + plan_phase + answer_phase
        input_set.append(input_sequence)

    return input_set

def gen_target_set(dataset):
    """
    Encodes the shortest paths from the dataset to their one-hot equivalents 
    """
    target_set = []
    for path in dataset[2]:
        encoded_path = []
        for edge in path:
            encoded_path.append(decimal_to_onehot(edge[0]) + decimal_to_onehot(edge[1]))
        target_set.append(encoded_path)

    return target_set
    