from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from dto.ErrorDTO import ErrorPercentage


def run_error_metrics(predictions, data):
    error = ErrorPercentage(data)
    error = run_mean_square_error(predictions, data.user_item_ratings_for_predict, data.n_users, data.n_items, error)
    error = run_mean_absolute_error(predictions, data.user_item_ratings_for_predict, data.n_users, data.n_items, error)

    return error


def run_mean_absolute_error(predictions, user_item_ratings, n_users, n_items, error):
    print(" ")
    print("                    Mean Absolute Error Results: ")
    error.pearson_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_ratings_by_pearson, n_users,
                                             n_items,
                                             "Predicted Ratings by Pearson   ")
    error.kmeans_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_ratings_by_kmeans, n_users,
                                            n_items,
                                            "Predicted Ratings by Kmeans    ")
    error.ACO_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_ratings_by_aco, n_users, n_items,
                                         "Predicted Ratings by ACO       ")
    error.ACO_Kmeans_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_ratings_by_aco_kmeans, n_users,
                                                n_items,
                                                "Predicted Ratings by ACO-Kmeans")
    error.age_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_rating_by_age, n_users, n_items,
                                         "Predicted Ratings by Age       ")
    error.sex_MAE = mean_absoluted_error(user_item_ratings, predictions.predicted_rating_by_sex, n_users, n_items,
                                         "Predicted Ratings by Sex       ")

    return error


def run_mean_square_error(predictions, user_item_ratings, n_users, n_items, error):
    print("                    Mean Square Error Results: ")
    error.pearson_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_pearson, n_users, n_items,
                                          "Predicted Ratings by Pearson    ")
    error.kmeans_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_kmeans, n_users, n_items,
                                         "Predicted Ratings by Kmeans     ")
    error.ACO_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_aco, n_users, n_items,
                                      "Predicted Ratings by ACO Cluster")
    error.ACO_Kmeans_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_aco_kmeans, n_users,
                                             n_items,
                                             "Predicted Ratings by ACO-Kmeans ")
    error.age_MSE = mean_square_error(user_item_ratings, predictions.predicted_rating_by_age, n_users, n_items,
                                      "Predicted Ratings by Age        ")
    error.sex_MSE = mean_square_error(user_item_ratings, predictions.predicted_rating_by_sex, n_users, n_items,
                                      "Predicted Ratings by Sex        ")

    error.pearson_RMSE = math.sqrt(error.pearson_MSE)
    error.kmeans_RMSE = math.sqrt(error.kmeans_MSE)
    error.ACO_RMSE = math.sqrt(error.ACO_MSE)
    error.ACO_Kmeans_RMSE = math.sqrt(error.ACO_Kmeans_MSE)
    error.age_RMSE = math.sqrt(error.age_MSE)
    error.sex_RMSE = math.sqrt(error.sex_MSE)

    print("\n")
    print("                    Root Mean Square Error Results: ")
    print("Predicted Ratings by Pearson     RMSE: ", error.pearson_RMSE)
    print("Predicted Ratings by Kmeans      RMSE: ", error.kmeans_RMSE)
    print("Predicted Ratings by ACO-Cluster RMSE: ", error.ACO_RMSE)
    print("Predicted Ratings by ACO-Kmeans  RMSE: ", error.ACO_Kmeans_RMSE)
    print("Predicted Ratings by Age         RMSE: ", error.age_RMSE)
    print("Predicted Ratings by Sex         RMSE: ", error.sex_RMSE)

    return error


def mean_square_error(test, utility, n_users, n_items, calculation_type):
    # test datası ile tehmin arasında MSE
    y_true = []
    y_pred = []
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                y_true.append(test[i][j])
                y_pred.append(utility[i][j])

    MSE = mean_squared_error(y_true, y_pred)
    print(calculation_type, " MSE: %f" % MSE)

    return MSE


def mean_absoluted_error(test, utility, n_users, n_items, calculation_type):
    # test datası ile tehmin arasında MAE
    y_true = []
    y_pred = []
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                y_true.append(test[i][j])
                y_pred.append(utility[i][j])

    MAE = mean_absolute_error(y_true, y_pred)
    print(calculation_type, " MAE: %f" % MAE)

    return MAE
