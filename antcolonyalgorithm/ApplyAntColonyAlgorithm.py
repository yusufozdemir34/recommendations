from antcolonyalgorithm import antcolony
from antcolonyalgorithm.antgraph import AntGraph
import random
import pants
import acopy
import tsplib95
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
import math
from sko.ACA import ACA_TSP

class AntColonyHelper:
    def ant_colony_by_acopy(n_users, pcs_matrix):
        num_points = n_users


        def cal_total_distance(routine):
            num_points, = routine.shape
            return sum([pcs_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


        aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                      size_pop=11, max_iter=11,
                      distance_matrix=pcs_matrix)

        best_x, best_y = aca.run()

        return aca.Tau

    # clust with ant colony optimization
    def ant_colony_optimization(n_users, pcs_matrix):
        graph = AntGraph(n_users, pcs_matrix)
        graph.reset_tau()
        num_iterations = 5
        # n_users = 5
        ant_colony = antcolony.AntColony(graph, 5, num_iterations)
        ant_colony.start()

        return graph.tau_mat
