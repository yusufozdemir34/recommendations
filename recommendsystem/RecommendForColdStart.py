from recommendsystem.RecommendationHelper import createCluster, cluster_means, create_avg_user, \
    create_model, set_one_for_max_avg_value_others_zero, get_prediction, mean_square_error
from antcolonyalgorithm.ant_colony_helper import AntColonyHelper
from ccl.ccl import connected_component_labelling
import numpy as np


class ColdStartRecommendation:
    def run_recommendation(self):
        # data arrays
        # find pcs_matrix relative to pearson algorithm (similarity matrix)
        user, item, test, pcs_matrix, utility, n_users, n_items = create_model()

        # Apply ant colony optimization to the similarity matrix
        result = AntColonyHelper.ant_colony_optimization(n_users, pcs_matrix)
        result1 = np.array(result)  # result2 = np.asmatrix(result1)
        # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
        result = set_one_for_max_avg_value_others_zero(result1)

        # tagging (labeling) for cluster
        result = connected_component_labelling(result, 4)
        clusterUser = []
        clusterUser = createCluster(result)
        # KNNalgorithm.getKNNalgorithm(clusterUser,1,1,1)

        # calculate average of each line (user)
        means = cluster_means(utility, clusterUser)
        # Calculate average value for user ratings and add to user object
        user = create_avg_user(user, n_users, utility)

        utility_copy = get_prediction(utility, pcs_matrix, user, clusterUser)

        # test datası ile tehmin arasında MSE
        mean_square_error(test, utility_copy, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
