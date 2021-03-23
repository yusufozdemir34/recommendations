from service.ModelService import prepare_data
from service.ClusterService import create_clusters_by_aco, create_clusters_by_kmeans, create_clusters_by_pearson
from service.AverageService import create_averages
from service.PredictionService import get_predictions
from service.CalculateErrorService import run_error_metrics


# we have generally five parts. Model, cluster, average, prediction, calculating error
class ColdStartRecommendation:
    def run_recommendation(self):
        # first part is model part
        user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items = prepare_data()

        # 3. Part is  average calculation part
        user, averages_ratings_by_demographics = create_averages(user, n_users, user_item_ratings_for_predict, n_items)

        # 2. Part is  cluster part
        clusters_by_pearson, pearson_average_ratings = create_clusters_by_pearson(user_item_ratings, user_user_pearson)
        clusters_by_aco = create_clusters_by_aco(n_users, user_user_pearson)
        clusters_by_kmeans, kmeans_avg, average_ratings_for_item_kmeans = create_clusters_by_kmeans(user_item_ratings, n_users, n_items)

        # 4. Part is predictions part
        predictions = get_predictions(user_item_ratings_for_predict, user, clusters_by_kmeans, kmeans_avg,average_ratings_for_item_kmeans,
                                      clusters_by_aco, averages_ratings_by_demographics, clusters_by_pearson,
                                      pearson_average_ratings)

        # 5. Part is error calculation part
        run_error_metrics(predictions, user_item_ratings, n_users, n_items)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
