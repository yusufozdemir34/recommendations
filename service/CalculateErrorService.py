from sklearn.metrics import mean_squared_error


def calcuate_mean_error(predictions, user_item_ratings, n_users, n_items):
    mean_square_error(user_item_ratings, predictions.predicted_ratings_by_pearson, n_users, n_items, "predicted_ratings_by_pearson")
    mean_square_error(user_item_ratings, predictions.predicted_ratings_by_kmeans, n_users, n_items, "predicted_ratings_by_kmeans")
    mean_square_error(user_item_ratings, predictions.predicted_ratings_by_aco, n_users, n_items, "predicted_ratings_by_aco")
    mean_square_error(user_item_ratings, predictions.predicted_rating_by_age, n_users, n_items, "predicted_rating_by_age")
    mean_square_error(user_item_ratings, predictions.predicted_rating_by_sex, n_users, n_items, "predicted_rating_by_sex")


def mean_square_error(test, utility, n_users, n_items, calculation_type):
    # test datası ile tehmin arasında MSE
    y_true = []
    y_pred = []
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                y_true.append(test[i][j])
                y_pred.append(utility[i][j])

    print(calculation_type, "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))
