class Prediction:
    def __init__(self, predicted_ratings_by_pearson, predicted_ratings_by_kmeans, predicted_ratings_by_aco, predicted_ratings_by_aco_kmeans,
                 predicted_rating_by_age,
                 predicted_rating_by_sex):
        self.predicted_ratings_by_pearson = predicted_ratings_by_pearson
        self.predicted_ratings_by_kmeans = predicted_ratings_by_kmeans
        self.predicted_ratings_by_aco = predicted_ratings_by_aco
        self.predicted_ratings_by_aco_kmeans = predicted_ratings_by_aco_kmeans
        self.predicted_rating_by_age = predicted_rating_by_age
        self.predicted_rating_by_sex = predicted_rating_by_sex
