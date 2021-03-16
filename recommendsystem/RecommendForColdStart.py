from clustering.FindClusterInMatrix import find_cluster_in_matrix
from recommendsystem.RecommendationHelper import create_avg_user, \
    set_one_for_max_avg_value_others_zero, get_prediction, mean_square_error, \
    cluster_mean_from_components, create_avg_ratings
from antcolonyalgorithm.ApplyAntColonyAlgorithm import AntColonyHelper
from recommendsystem.FileHelper import load_model_as_np, create_model
import numpy as np


class ColdStartRecommendation:
    def run_recommendation(self):
        # data arrays
        # find pcs_matrix relative to pearson algorithm (similarity matrix)
        utility, test, user, item, pcs_matrix = load_model_as_np()
        n_users = len(user)
        n_items = len(item)

        for i in range(0, n_users):
            for j in range(0, n_items):
                if utility[i][j] != test[i][j]:
                    utility[i][j] = -1
                    print("-1 yapilcaklar i: ", i, " j: ", j)
        # Apply ant colony optimization to the similarity matrix
        # ant_colony_user_cluster = AntColonyHelper.ant_colony_by_acopy(n_users, pcs_matrix)
        # np.save("../data/ant_colony_by_acopy.npy", ant_colony_user_cluster)
        ant_colony_user_cluster = np.load("../data/runned_data/ant_colony_by_acopy.npy")
        result = np.array(ant_colony_user_cluster)
        # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
        result = set_one_for_max_avg_value_others_zero(result)

        clustered_users, nxg, components = find_cluster_in_matrix(result, n_users)

        means = cluster_mean_from_components(utility, clustered_users)

        # Calculate average value for user ratings and add to user object
        user = create_avg_user(user, n_users, utility)

        avg = create_avg_ratings(user, n_users, utility, n_items)

        predicted_ratings, rating_by_age, rating_by_sex = get_prediction(utility, pcs_matrix, user, clustered_users, avg)

        # test datası ile tehmin arasında MSE
        mean_square_error(test, predicted_ratings, n_users, n_items)
        mean_square_error(test, rating_by_age, n_users, n_items)
        mean_square_error(test, rating_by_sex, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
