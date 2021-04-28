class ErrorPercentage:
    def __init__(self, clusters_by_pearson):
        self.pearson_MSE = 0
        self.kmeans_MSE = 0
        self.ACO_MSE = 0
        self.ACO_Kmeans_MSE = 0
        self.age_MSE = 0
        self.sex_MSE = 0

        self.pearson_RMSE = 0
        self.kmeans_RMSE = 0
        self.ACO_RMSE = 0
        self.ACO_Kmeans_RMSE = 0
        self.age_RMSE = 0
        self.sex_RMSE = 0

        self.pearson_MAE = 0
        self.kmeans_MAE = 0
        self.ACO_MAE = 0
        self.ACO_Kmeans_MAE = 0
        self.age_MAE = 0
        self.sex_MAE = 0
