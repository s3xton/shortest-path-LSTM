"""
Graph utility class for generating random graph data, finding shortest paths, and formatting
the data for input to the network
"""
import math
import random
import dataset
import graph as gr

def gen_graph_data(set_size, min_node, max_node):
    """
    Generates random graph data to be used for training.

    Args:
        set_size: number of elements in the set
        min_node: the minimum number of nodes in any graph in the set
        max_node: the maximum number of nodes in any graph in the set

    Returns:
        graph_data: a list of Graph objects
    """
    graph_data = []
    for _ in range(0, set_size):
        graph_size = random.randint(min_node, max_node)
        graph_data.append(gr.Graph(graph_size))

    return graph_data


def dijkstra(graph, start):
    """
    Implementation of Dijkstra's algorithm that finds the distances
    of the shortest paths between a start node and every other node in the graph.

    Args:
        graph: the graph to search
        start: the start node of the paths

    Returns:
        dist: a list of distances from the start node to each other node.
        prev: the previous node for each node in the graph.
    """
    nodes = graph.nodes
    adj = graph.adjacency
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
    Finds the shortest path between given start and end nodes of a graph
    using Dijkstra's algorithm.

    Args:
        graph: the graph to search.
        start: the start node.
        end: the end node.

    Returns:
        path_as_edge_list: the shortest path between start and end in graph, defined as
            a list of edges

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

    Args:
        graph_data: a list of Graph objects

    Returns:
        terminal_pairs: a list of tuples corresponding to the start and end nodes of the
            shortest paths found.
        shortest_paths: the shortest paths between the terminal nodes specified as a list
            of edges
    """
    terminal_pairs = []
    shortest_paths = []
    path_lengths = []
    for i in range(0, len(graph_data)):
        graph = graph_data[i]
        terminal_pairs.append(random.sample(graph.nodes, 2))
        path = shortest_path(graph, terminal_pairs[i][0], terminal_pairs[i][1])
        shortest_paths.append(path)
        path_lengths.append(len(path))

    return terminal_pairs, shortest_paths, path_lengths

def build_dataset(set_size, min_node, max_node):
    """
    Builds the entire dataset.

    Args:
        set_size: number of elements in the set
        min_node: the minimum number of nodes in any graph in the set
        max_node: the maximum number of nodes in any graph in the set

    Returns:
        dataset: a dataset object
    """

    graph_list = gen_graph_data(set_size, min_node, max_node)
    terminal_nodes, shortest_paths, path_lengths = gen_shortest_paths(graph_list)

    return dataset.Dataset(graph_list,
                           terminal_nodes,
                           shortest_paths,
                           path_lengths,
                           min_node,
                           max_node)
    