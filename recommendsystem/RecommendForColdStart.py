from clustering.FindClusterInMatrix import find_cluster_in_matrix
from recommendsystem.RecommendationHelper import createCluster, cluster_means, create_avg_user, \
    set_one_for_max_avg_value_others_zero, get_prediction, mean_square_error, \
    cluster_mean_from_components
from antcolonyalgorithm.ant_colony_helper import AntColonyHelper
from recommendsystem.FileHelper import create_model, load_model_as_np
import numpy as np


class ColdStartRecommendation:
    def run_recommendation(self):
        # data arrays
        # find pcs_matrix relative to pearson algorithm (similarity matrix)
        user, item, test, pcs_matrix, utility, n_users, n_items = create_model()

        # Apply ant colony optimization to the similarity matrix
        # ant_colony_user_cluster = np.load("../data/aop_clustering_result.npy")
        ant_colony_user_cluster = AntColonyHelper.ant_colony_optimization(943, pcs_matrix)
        # np.save("../data/aop_clustering_result.npy", antColony)
        result = np.array(ant_colony_user_cluster)
        # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
        result = set_one_for_max_avg_value_others_zero(result)

        # tagging (labeling) for cluster
        # result = connected_component_labelling(result, 4)
        clustered_users, nxg, components = find_cluster_in_matrix(result, n_users)
        # for nodes in clusterUser:
        #     print(nodes)
        means = cluster_mean_from_components(utility, clustered_users)
        # clusterUser = createCluster(result)
        # KNNalgorithm.getKNNalgorithm(clusterUser,1,1,1)
        # clusterUser = np.array(clusterUser)
        # calculate average of each line (user)
        # means = cluster_means(utility, clusterUser)
        # Calculate average value for user ratings and add to user object
        user = create_avg_user(user, n_users, utility)

        utility_copy = get_prediction(utility, pcs_matrix, user, clustered_users)

        # test datası ile tehmin arasında MSE
        mean_square_error(test, utility, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
