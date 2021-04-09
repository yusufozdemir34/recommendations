from service.ModelService import prepare_data
from service.ClusterService import create_clusters_by_aco, create_clusters_by_kmeans, create_clusters_by_pearson, \
    create_clusters
from service.AverageService import create_averages
from service.PredictionService import get_predictions
from service.CalculateErrorService import run_error_metrics


# we have generally five parts. Model, cluster, average, prediction, calculating error
class ColdStartRecommendation:
    def run_recommendation(self):
        # first part is model part
        dataDTO = prepare_data()

        # 2. Part is  cluster part
        clusters, dataDTO = create_clusters(dataDTO)

        # 3. Part is  average calculation part
       # dataDTO.user, clusters.averages_ratings_by_demographics = create_averages(dataDTO)

        # 4. Part is predictions part
        predictions = get_predictions(dataDTO, clusters)

        # 5. Part is error calculation part
        run_error_metrics(predictions, dataDTO)


if __name__ == '__main__':
    ColdStartRecommendation.run_recommendation(19)
