import numpy as np
from sklearn.cluster import KMeans
# clear data for clustering
from clustering.FindClusterInMatrix import find_cluster_in_matrix_by_nx


def create_clusters_by_aco(n_users, user_user_pearson):
    # Apply ant colony optimization to the similarity matrix
    # ant_colony_user_cluster = AntColonyHelper.ant_colony_by_acopy(n_users, pcs_matrix)
    # np.save("../data/ant_colony_by_acopy.npy", ant_colony_user_cluster)
    user_clusters_by_aco = np.load("../data/runned_data/ant_colony_by_acopy.npy")
    result = np.array(user_clusters_by_aco)
    # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
    result = set_one_for_max_avg_value_others_zero(result)
    clustered_users = find_cluster_in_matrix(result, n_users)

    return clustered_users


def cluster_by_kmeans(matris):
    clusters = KMeans(n_clusters=5).fit_predict(matris)

    return clusters


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
