from service.ModelService import prepare_data
from service.ClusterService import create_clusters_by_aco
from service.AverageService import create_averages
from service.PredictionService import get_predictions
from service.CalculateErrorService import calcuate_mean_error


# we have generally five parts. Model, cluster, average, prediction, calculating error
class ColdStartRecommendation:
    def run_recommendation(self):
        # first part is model part
        user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items = prepare_data()

        # 2. Part is  cluster part
        clustered_users = create_clusters_by_aco(n_users, user_user_pearson)
        # clusters_by_kmeans = cluster_by_kmeans(user_item_ratings_for_predict)

        # 3. Part is  average calculation part
        user, avg = create_averages(user, n_users, user_item_ratings_for_predict, n_items)

        # 4. Part is predictions part
        predictions = get_predictions(user_item_ratings_for_predict, user_user_pearson, user, clustered_users, avg)

        # 5. Part is error calculation part
        calcuate_mean_error(predictions, user_item_ratings, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
