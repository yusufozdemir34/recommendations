from sklearn.metrics import mean_squared_error
import math


def run_error_metrics(predictions, user_item_ratings, n_users, n_items):
    run_mean_square_error(predictions, user_item_ratings, n_users, n_items)


def run_mean_square_error(predictions, user_item_ratings, n_users, n_items):
    print("                    Mean Square Error Results: ")
    pearson_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_pearson, n_users, n_items,
                                    "Predicted Ratings by Pearson")
    kmeans_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_kmeans, n_users, n_items,
                                   "Predicted Ratings by Kmeans ")
    ACO_MSE = mean_square_error(user_item_ratings, predictions.predicted_ratings_by_aco, n_users, n_items,
                                "Predicted Ratings by ACO    ")
    age_MSE = mean_square_error(user_item_ratings, predictions.predicted_rating_by_age, n_users, n_items,
                                "Predicted Ratings by Age    ")
    sex_MSE = mean_square_error(user_item_ratings, predictions.predicted_rating_by_sex, n_users, n_items,
                                "Predicted Ratings by Sex    ")

    pearson_RMSE = math.sqrt(pearson_MSE)
    kmeans_RMSE = math.sqrt(kmeans_MSE)
    ACO_RMSE = math.sqrt(ACO_MSE)
    age_RMSE = math.sqrt(age_MSE)
    sex_RMSE = math.sqrt(sex_MSE)

    print("\n")
    print("                    Root Mean Square Error Results: ")
    print("Predicted Ratings by Pearson RMSE: ", pearson_RMSE)
    print("Predicted Ratings by Kmeans  RMSE: ", kmeans_RMSE)
    print("Predicted Ratings by ACO     RMSE: ", ACO_RMSE)
    print("Predicted Ratings by Age     RMSE: ", age_RMSE)
    print("Predicted Ratings by Sex     RMSE: ", sex_RMSE)


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
