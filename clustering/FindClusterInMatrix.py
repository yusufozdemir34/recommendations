import numpy as np
import networkx as nx
import numpy as np
from collections import defaultdict


class FindClusterInMatrix:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0


def find_cluster_in_matrix_by_nx(connected_point, n_cluster):
    connected = np.where(connected_point == 1)
    G = nx.Graph()
    user_nodes = np.arange(0, n_cluster)
    G.add_nodes_from(user_nodes)
    connected = np.column_stack(connected)
    connected = tuple(map(tuple, connected))
    G.add_edges_from(connected)
    components = nx.connected_components(G)
    result = []
    for nodes in components:
        temp = []
        for node in nodes:
            temp.append(node)
        result.append(temp)

    # result = get_result_from_component(components, n_cluster)

    return result, G, components


def get_result_from_component(components, n_cluster):
    result = []
    i = 0
    j = 0
    for nodes in components:
        temp = []
        for node in nodes:
            temp.append(node)
        result.append(temp)
        i = i + 1
    return result
