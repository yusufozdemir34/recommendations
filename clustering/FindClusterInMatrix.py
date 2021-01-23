import numpy as np
import networkx as nx
import numpy as np
from collections import defaultdict


def find_cluster_in_matrix(connected_point, n_user):
    connected_edge1, connected_edge2 = np.where(connected_point == 1)
    G = nx.Graph()
    user_nodes = np.arange(1, n_user)
    G.add_nodes_from(user_nodes)
    print(connected_edge1)
    print(connected_edge2)
    connected = [(int(connected_edge1), int(connected_edge2))]
    G.add_edges_from(connected)


connected_edges = np.asmatrix('2  2  0 ; 1  0  0; 0  0  0')
find_cluster_in_matrix(connected_edges, 3)
