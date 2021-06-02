from service.ModelService import prepare_data, create_model
from service.ClusterService import create_clusters_by_aco, create_clusters_by_kmeans, create_clusters_by_pearson, \
    create_clusters
from service.AverageService import create_averages, calculate_average_for_errors
from service.PredictionService import get_predictions
from service.CalculateErrorService import run_error_metrics


# we have generally five parts. Model, cluster, average, prediction, calculating error
class ColdStartRecommendation:
    def run_recommendation(self):
        # first part is model part
        dataDTO = create_model()

        # 2. Part is  cluster part
        clusters, dataDTO = create_clusters(dataDTO)

        # 3. Part is  average calculation part
        # dataDTO.user, clusters.averages_ratings_by_demographics = create_averages(dataDTO)

        # 4. Part is predictions part
        predictions = get_predictions(dataDTO, clusters)

        # 5. Part is error calculation part
        error = run_error_metrics(predictions, dataDTO)

        return error


if __name__ == '__main__':
    error1 = ColdStartRecommendation.run_recommendation(19)
    # error2 = ColdStartRecommendation.run_recommendation(19)
    # error3 = ColdStartRecommendation.run_recommendation(19)
    # error4 = ColdStartRecommendation.run_recommendation(19)
    # error5 = ColdStartRecommendation.run_recommendation(19)
    #
    # error = calculate_average_for_errors(error1, error2, error3, error4, error5)
