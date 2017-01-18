import random
import itertools
import numpy as np


class Graph:
    """A class that will generate a random connected graph
     of a specified size. Represented as an edge list.

     I decided to represent the graphs in terms of actual numbers and then format
     them later for input to the network, rather than to generate them in a format
     acceptable by the network (bit sequences) to begin with. This is because it
     is easier to manipulate and debug the graphs using integers.
     """
    def __init__(self, graph_size):
        self.nodes = list(range(graph_size))
        self.size = graph_size

        max_edges = graph_size * (graph_size - 1) / 2
        min_edges = graph_size - 1
        self.edge_number = random.randint(min_edges, max_edges)

        self.graph = []
        possible_edges = list(itertools.combinations(self.nodes, 2))

        # All graphs must be connected, therefore there is at least one path connecting
        # all nodes. Begin by forming this path first, at random.
        shuffled_nodes = self.nodes
        random.shuffle(shuffled_nodes)
        for i in range(0, min_edges):
            edge = (shuffled_nodes[i], shuffled_nodes[i + 1])
            self.graph.append(edge)

            # Little bit inefficient having to check both orderings of the edge...
            # Unavoidable?...
            if edge in possible_edges:
                possible_edges.remove(edge)
            else:
                possible_edges.remove(edge[::-1])

        # For many graphs, there will be more edges, add these now. Again maintaining
        # randomness in construction.
        remaining_edges = self.edge_number - min_edges
        for _ in range(0, remaining_edges):
            self.graph.append(random.choice(possible_edges))

        # Shuffle the graph so that there are definitely no patterns.
        random.shuffle(self.graph)
        self.__set_adjacency()

    def __set_adjacency(self):
        # Create an adjacency matrix to be used for faster lookup for shortest paths
        self.adjacency = []
        for i in range(0, self.size):
            self.adjacency.append([0] * self.size)

        for edge in self.graph:
            self.adjacency[edge[0]][edge[1]] = 1
            self.adjacency[edge[1]][edge[0]] = 1

    def set_graph(self, nodes, edge_list):
        """Used to explicitly define graphs for debugging purposes"""
        self.nodes = nodes
        self.graph = edge_list
        self.size = len(nodes)
        self.edge_number = len(edge_list)
        self.__set_adjacency()

    def get_adjacency(self):
        return self.adjacency

    def get_graph(self):
        return self.graph

    def get_size(self):
        return self.size

    def get_edge_number(self):
        return self.edge_number

    def get_nodes(self):
        return self.nodes

    def to_string(self):
        return self.graph

