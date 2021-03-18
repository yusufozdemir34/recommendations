import numpy as np
from service.ModelService import load_model_as_np, create_model
from service.ClusterService import set_one_for_max_avg_value_others_zero, cluster_mean_from_components, \
    find_cluster_in_matrix
from service.AverageService import create_avg_user, create_avg_ratings
from service.PredictionService import get_predictions
from service.CalculateErrorService import calcuate_mean_error


# we have generally five parts. Model, cluster, average, prediction, calculating error
class ColdStartRecommendation:
    def run_recommendation(self):
        # first part is model part
        # find user_user_pearson relative to pearson algorithm
        user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson = load_model_as_np()
        n_users = len(user)
        n_items = len(item)

        for i in range(0, n_users):
            for j in range(0, n_items):
                if user_item_ratings_for_predict[i][j] != user_item_ratings[i][j]:
                    user_item_ratings_for_predict[i][j] = -1
                    print("-1 yapilcaklar i: ", i, " j: ", j)

        # 2. Part is  cluster part
        # Apply ant colony optimization to the similarity matrix
        # ant_colony_user_cluster = AntColonyHelper.ant_colony_by_acopy(n_users, pcs_matrix)
        # np.save("../data/ant_colony_by_acopy.npy", ant_colony_user_cluster)
        user_clusters_by_aco = np.load("../data/runned_data/ant_colony_by_acopy.npy")
        result = np.array(user_clusters_by_aco)
        # Assign values 1 and 0 to disable places that some ants use for exploration and could find less or nothing.
        result = set_one_for_max_avg_value_others_zero(result)
        clustered_users = find_cluster_in_matrix(result, n_users)

        # 3. Part is  average calculation part
        # Calculate average value for user ratings and add to user object
        user = create_avg_user(user, n_users, user_item_ratings_for_predict)
        avg = create_avg_ratings(user, n_users, user_item_ratings_for_predict, n_items)

        # 4. Part is predictions part
        predictions = get_predictions(user_item_ratings_for_predict, user_user_pearson, user, clustered_users, avg)

        # 5. Part is error calculation part
        calcuate_mean_error(predictions, user_item_ratings, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
