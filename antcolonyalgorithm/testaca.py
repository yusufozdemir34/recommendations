import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
import math

from service.ModelService import load_model_as_np

num_points = 943
# find pcs_matrix relative to pearson algorithm (similarity matrix)
user, item, test, pcs_matrix, utility, n_users, n_items = load_model_as_np()

try:
    for i in range(0, n_users):
        for j in range(0, n_users):
            if math.isnan(pcs_matrix[i][j]):
                pcs_matrix[i][j]=0.2
except:
    print("An exception occurred")
points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(pcs_matrix, pcs_matrix, metric='euclidean')


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([pcs_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA
from sko.ACA import ACA_TSP



aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=5, max_iter=5,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

# %% Plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
