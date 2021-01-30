from clustering.FindClusterInMatrix import find_cluster_in_matrix
from recommendsystem.RecommendationHelper import createCluster, cluster_means, create_avg_user, \
    create_model, set_one_for_max_avg_value_others_zero, get_prediction, mean_square_error, \
    cluster_mean_from_components, load_model_as_np
from antcolonyalgorithm.ant_colony_helper import AntColonyHelper
from ccl.ccl import connected_component_labelling
import numpy as np


class ColdStartRecommendation:
    def run_recommendation(self):
        # data arrays
        # find pcs_matrix relative to pearson algorithm (similarity matrix)
        user, item, test, pcs_matrix, utility, n_users, n_items = load_model_as_np()

        # Apply ant colony optimization to the similarity matrix
        result = np.load("../data/aop_clustering_result.npy")
        # result = AntColonyHelper.ant_colony_optimization(n_users, pcs_matrix)
        # np.save("../data/aop_clustering_result", result)
        result = np.array(result)
        # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
        result = set_one_for_max_avg_value_others_zero(result)

        # tagging (labeling) for cluster
        # result = connected_component_labelling(result, 4)
        clusterUser = find_cluster_in_matrix(result, n_users)
        for nodes in clusterUser:
            print(nodes)
        means = cluster_mean_from_components(utility, clusterUser)
        # clusterUser = createCluster(result)
        # KNNalgorithm.getKNNalgorithm(clusterUser,1,1,1)
        # clusterUser = np.array(clusterUser)
        # calculate average of each line (user)
        # means = cluster_means(utility, clusterUser)
        # Calculate average value for user ratings and add to user object
        user = create_avg_user(user, n_users, utility)

        utility_copy = get_prediction(utility, pcs_matrix, user, clusterUser)

        # test datası ile tehmin arasında MSE
        mean_square_error(test, utility_copy, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
