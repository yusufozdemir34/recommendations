import numpy as np
from sklearn.cluster import KMeans
# clear data for clustering
from antcolonyalgorithm.ApplyAntColonyAlgorithm import AntColonyHelper
from clustering.FindClusterInMatrix import find_cluster_in_matrix_by_nx
from service.AverageService import calculate_avg_for_kmeans, calculate_avg_for_pearson


def create_clusters_by_user_user_pearson(user_user_pearson):
    user_smilarity_top_list = np.zeros((943, 5))

    smilarity_value5 = 0

    for j in range(0, 942):
        user1 = 1
        user2 = 2
        user3 = 3
        user4 = 4
        user5 = 5
        smilarity_value1 = user_user_pearson[1][j]
        smilarity_value2 = user_user_pearson[2][j]
        smilarity_value3 = user_user_pearson[3][j]
        smilarity_value4 = user_user_pearson[4][j]

        for i in range(1, 942):  # sorting big to small
            if smilarity_value1 <= user_user_pearson[i][j]:
                user1 = i
                smilarity_value2 = smilarity_value1
                smilarity_value3 = smilarity_value2
                smilarity_value4 = smilarity_value3
                smilarity_value5 = smilarity_value4
                smilarity_value1 = user_user_pearson[i][j]
            elif smilarity_value2 <= user_user_pearson[i][j]:
                user2 = i
                smilarity_value3 = smilarity_value2
                smilarity_value4 = smilarity_value3
                smilarity_value5 = smilarity_value4
                smilarity_value2 = user_user_pearson[i][j]
            elif smilarity_value3 <= user_user_pearson[i][j]:
                user3 = i
                smilarity_value4 = smilarity_value3
                smilarity_value5 = smilarity_value4
                smilarity_value3 = user_user_pearson[i][j]
            elif smilarity_value4 <= user_user_pearson[i][j]:
                user4 = i
                smilarity_value5 = smilarity_value4
                smilarity_value4 = user_user_pearson[i][j]
            elif smilarity_value5 <= user_user_pearson[i][j]:
                user5 = i
                smilarity_value5 = user_user_pearson[i][j]

        user_smilarity_top_list[j][0] = user1
        user_smilarity_top_list[j][1] = user2
        user_smilarity_top_list[j][2] = user3
        user_smilarity_top_list[j][3] = user4
        user_smilarity_top_list[j][4] = user5
        total_ratings = 0
        count_ratings = 0

    return user_smilarity_top_list


def create_clusters_by_pearson(ratings, user_user_pearson):
    user_smilarity_top_list = create_clusters_by_user_user_pearson(user_user_pearson)
    smilar_user_average_ratings = calculate_avg_for_pearson(user_smilarity_top_list, ratings, 943, 1682)

    return user_smilarity_top_list, smilar_user_average_ratings


def create_clusters_by_aco(n_users, user_user_pearson):
    # Apply ant colony optimization to the similarity matrix
    # user_clusters_by_aco = AntColonyHelper.ant_colony_by_acopy(n_users, user_user_pearson)
    # np.save("../data/runned_data/user_clusters_by_aco.npy", user_clusters_by_aco)
    user_clusters_by_aco = np.load("../data/runned_data/user_clusters_by_aco.npy")



    result = np.array(user_clusters_by_aco)
    # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
    result = set_one_for_max_avg_value_others_zero(result)
    clustered_users = find_cluster_in_matrix(result, n_users)

    return clustered_users


def create_clusters_by_kmeans(ratings, n_users, n_items):
    clusters_by_kmeans = KMeans(n_clusters=5).fit_predict(ratings)
    avg, average_ratings_for_items = calculate_avg_for_kmeans(ratings, clusters_by_kmeans, n_users, n_items)
    return clusters_by_kmeans, avg, average_ratings_for_items


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
