import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
# clear data for clustering
from antcolonyalgorithm.ApplyAntColonyAlgorithm import AntColonyHelper
from clustering.FindClusterInMatrix import find_cluster_in_matrix_by_nx
from dto.ClusterDTO import Clusters
from geneticalgorithm.ApplyGeneticAlgorithm import GeneticAlgorithm
from service.AverageService import calculate_avg_for_kmeans, calculate_avg_for_pearson, create_averages


def create_clusters(data):
    clusters_by_pearson, pearson_average_ratings = create_clusters_by_pearson(data.user_item_ratings,
                                                                              data.user_user_pearson)
    tau_by_aco = create_tau_by_aco(data.n_users, data.user_user_pearson)
    clusters_by_aco, average_ratings_for_aco = create_clusters_by_aco(data.n_users, tau_by_aco)
    clusters_by_aco_kmeans, average_ratings_for_item_aco = create_clusters_by_aco_kmeans(data.user_item_ratings,
                                                                                         data.n_users, data.n_items,
                                                                                         tau_by_aco)

    clusters_by_kmeans, average_ratings_for_item_kmeans = create_clusters_by_kmeans(
        data.user_item_ratings, data.n_users, data.n_items)

    clusters = Clusters(clusters_by_pearson, pearson_average_ratings, clusters_by_aco, clusters_by_aco_kmeans,
                        average_ratings_for_item_aco, clusters_by_kmeans,
                        average_ratings_for_item_kmeans)

    data.user, clusters.average_ratings_for_age_by_items, clusters.average_ratings_for_sex_by_items = create_averages(
        data)

    return clusters, data


def normalize(matrix_data):
    # scaler = preprocessing.StandardScaler().fit(matrix_data)
    # X_scaled = scaler.transform(matrix_data)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(matrix_data)

    return X_train_minmax


def create_clusters_by_pearson(ratings, user_user_pearson):
    user_smilarity_top_list = create_clusters_by_user_user_pearson(user_user_pearson)
    smilar_user_average_ratings = calculate_avg_for_pearson(user_smilarity_top_list, ratings, np.size(ratings, 0),
                                                            np.size(ratings, 1))

    return user_smilarity_top_list, smilar_user_average_ratings


def create_clusters_by_ga(n_users, user_user_pearson):
    user_clusters_by_ga = GeneticAlgorithm.cluster_by_genetic_algoritm(n_users, user_user_pearson)
    result = np.array(user_clusters_by_ga)
    result = set_one_for_max_avg_value_others_zero(result)
    clustered_users = find_cluster_in_matrix(result, n_users)


def create_clusters_by_aco_kmeans(ratings, n_users, n_items, tau_by_aco):
    clusters_by_aco_kmeans = KMeans(n_clusters=5).fit_predict(tau_by_aco)
    average_ratings_for_items = calculate_avg_for_kmeans(ratings, clusters_by_aco_kmeans, n_users, n_items)

    return clusters_by_aco_kmeans, average_ratings_for_items


def create_tau_by_aco(n_users, user_user_pearson):
    # Apply ant colony optimization to the similarity matrix
    # norm_data = normalize(user_user_pearson)
    tau_by_aco = AntColonyHelper.ant_colony_by_acopy(user_user_pearson)
    # np.save("../data/runned_data/user_clusters_by_aco.npy", user_clusters_by_aco)
    # tau_by_aco = np.load("../data/runned_data/user_clusters_by_aco.npy")
    return tau_by_aco


def create_clusters_by_aco(n_users, user_clusters_by_aco):
    average_ratings_for_aco = 0
    result = np.array(user_clusters_by_aco)
    # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
    result = set_one_for_max_avg_value_others_zero(result)
    clustered_users = find_cluster_in_matrix(result, n_users)

    return clustered_users, average_ratings_for_aco


def create_clusters_by_kmeans(ratings, n_users, n_items):
    clusters_by_kmeans = KMeans(n_clusters=5).fit_predict(ratings)
    average_ratings_for_items = calculate_avg_for_kmeans(ratings, clusters_by_kmeans, n_users, n_items)

    return clusters_by_kmeans, average_ratings_for_items


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


def create_clusters_by_user_user_pearson(user_user_pearson):
    user_smilarity_top_list = np.zeros((np.size(user_user_pearson, 0), 5))

    smilarity_value5 = 0

    for j in range(0, np.size(user_user_pearson, 0)):
        user1 = 1
        user2 = 2
        user3 = 3
        user4 = 4
        user5 = 5
        smilarity_value1 = user_user_pearson[1][j]
        smilarity_value2 = user_user_pearson[2][j]
        smilarity_value3 = user_user_pearson[3][j]
        smilarity_value4 = user_user_pearson[4][j]

        for i in range(1, np.size(user_user_pearson, 1)):  # sorting big to small
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
