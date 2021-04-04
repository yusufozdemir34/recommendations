from sko.GA import GA_TSP
import numpy as np
from scipy import spatial


class GeneticAlgorithm:
    def cluster_by_genetic_algoritm(n_users, pcs_matrix):
        num_points = n_users

        distance_matrix = spatial.distance.cdist(pcs_matrix, pcs_matrix, metric='euclidean')

        def cal_total_distance(routine):
            '''The objective function. input routine, return total distance.
            cal_total_distance(np.arange(num_points))
            '''
            num_points, = routine.shape
            return sum(
                [distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

        ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
        best_points, best_distance = ga_tsp.run()
        print(best_points)
        return ga_tsp
