import numpy as np

# clear data for clustering
from clustering.FindClusterInMatrix import find_cluster_in_matrix_by_nx


def set_one_for_max_avg_value_others_zero(data):
    max_matrix = data.max(0)
    avg_matrix = np.mean(max_matrix)
    for_i_size = np.size(data, 1)
    for_y_size = np.size(data, 0)

    # delta_math ın içindeki en yüksek avg yi bul. sonra en yüksek avg ye bir de. diğerleri sıfırdır.
    for j in range(0, for_i_size):
        for i in range(0, for_y_size):
            if data[i][j] == max_matrix[j] and max_matrix[j] > avg_matrix * 0.92:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


def find_cluster_in_matrix(connected_point, n_cluster):
    clustered_users, nxg, components = find_cluster_in_matrix_by_nx(connected_point, n_cluster)

    return clustered_users


def cluster_mean_from_components(utility, components):
    cluster_avg = []
    for i in range(0, len(utility)):
        cluster_avg.append(np.mean(utility[i]))

    return cluster_avg
    #
    # cluster_avg = []
    # # calculate average of each line (user)
    # for nodes in components:
    #     temp = []
    #     for node in nodes:
    #         temp.append(utility[node])
    #     cluster_avg.append(np.mean(temp))
    # return cluster_avg
