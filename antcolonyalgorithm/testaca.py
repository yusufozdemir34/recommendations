import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
import math

from antcolonyalgorithm.ApplyAntColonyAlgorithm import AntColonyHelper
from service.ClusterService import set_one_for_max_avg_value_others_zero, find_cluster_in_matrix
from service.ModelService import load_model_as_np

num_points = 943
# find pcs_matrix relative to pearson algorithm (similarity matrix)
# user, item, test, pcs_matrix, utility, n_users, n_items = load_model_as_np()
utility, test, user, item, user_user_pearson = load_model_as_np()
n_users = len(user)
# try:
#     for i in range(0, n_users):
#         for j in range(0, n_users):
#             if math.isnan(user_user_pearson[i][j]):
#                 user_user_pearson[i][j] = 0.2
# except:
#     print("An exception occurred")
# points_coordinate = np.random.rand(num_points, 943)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(user_user_pearson, user_user_pearson, metric='euclidean')


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([user_user_pearson[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA
from sko.ACA import ACA_TSP

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=2, max_iter=2,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

user_clusters_by_aco = AntColonyHelper.ant_colony_by_acopy(n_users, user_user_pearson)
result = np.array(user_clusters_by_aco)
# Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
result = set_one_for_max_avg_value_others_zero(result)
clustered_users = find_cluster_in_matrix(result, n_users)

# %% Plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = user_user_pearson[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
