class Clusters:
    def __init__(self, clusters_by_pearson, pearson_average_ratings, clusters_by_aco, clusters_by_kmeans, kmeans_avg,
                 average_ratings_for_item_kmeans):
        self.clusters_by_pearson = clusters_by_pearson
        self.pearson_average_ratings = pearson_average_ratings
        self.clusters_by_aco = clusters_by_aco
        self.clusters_by_kmeans = clusters_by_kmeans
        self.kmeans_avg = kmeans_avg
        self.average_ratings_for_item_kmeans = average_ratings_for_item_kmeans
