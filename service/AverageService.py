from domain.Average import Average
from domain.KMeansDomain import KMeansDomain
import numpy as np

from dto.ErrorDTO import ErrorPercentage


def calculate_average_for_errors(error1, error2, error3, error4, error5):
    error = ErrorPercentage(0)

    error.pearson_MSE = (error1.pearson_MSE + error2.pearson_MSE + error3.pearson_MSE + error4.pearson_MSE + error5.pearson_MSE) / 5
    error.kmeans_MSE = ( error1.kmeans_MSE + error2.kmeans_MSE + error3.kmeans_MSE + error4.kmeans_MSE + error5.kmeans_MSE) / 5
    error.ACO_MSE = (error1.ACO_MSE + error2.ACO_MSE + error3.ACO_MSE + error4.ACO_MSE + error5.ACO_MSE) / 5
    error.ACO_Kmeans_MSE = (error1.ACO_Kmeans_MSE + error2.ACO_Kmeans_MSE + error3.ACO_Kmeans_MSE + error4.ACO_Kmeans_MSE + error5.ACO_Kmeans_MSE) / 5
    error.age_MSE = (error1.age_MSE + error2.age_MSE + error3.age_MSE + error4.age_MSE + error5.age_MSE) / 5
    error.sex_MSE = (error1.sex_MSE + error2.sex_MSE + error3.sex_MSE + error4.sex_MSE + error5.sex_MSE) / 5

    error.pearson_RMSE = (error1.pearson_RMSE + error2.pearson_RMSE + error3.pearson_RMSE + error4.pearson_RMSE + error5.pearson_RMSE) / 5
    error.kmeans_RMSE = ( error1.kmeans_RMSE + error2.kmeans_RMSE + error3.kmeans_RMSE + error4.kmeans_RMSE + error5.kmeans_RMSE) / 5
    error.ACO_RMSE = (error1.ACO_RMSE + error2.ACO_RMSE + error3.ACO_RMSE + error4.ACO_RMSE + error5.ACO_RMSE) / 5
    error.ACO_Kmeans_RMSE = (error1.ACO_Kmeans_RMSE + error2.ACO_Kmeans_RMSE + error3.ACO_Kmeans_RMSE + error4.ACO_Kmeans_RMSE + error5.ACO_Kmeans_RMSE) / 5
    error.age_RMSE = (error1.age_RMSE + error2.age_RMSE + error3.age_RMSE + error4.age_RMSE + error5.age_RMSE) / 5
    error.sex_RMSE = (error1.sex_RMSE + error2.sex_RMSE + error3.sex_RMSE + error4.sex_RMSE + error5.sex_RMSE) / 5

    error.pearson_MAE = (error1.pearson_MAE + error2.pearson_MAE + error3.pearson_MAE + error4.pearson_MAE + error5.pearson_MAE) / 5
    error.kmeans_MAE = ( error1.kmeans_MAE + error2.kmeans_MAE + error3.kmeans_MAE + error4.kmeans_MAE + error5.kmeans_MAE) / 5
    error.ACO_MAE = (error1.ACO_MAE + error2.ACO_MAE + error3.ACO_MAE + error4.ACO_MAE + error5.ACO_MAE) / 5
    error.ACO_Kmeans_MAE = (error1.ACO_Kmeans_MAE + error2.ACO_Kmeans_MAE + error3.ACO_Kmeans_MAE + error4.ACO_Kmeans_MAE + error5.ACO_Kmeans_MAE) / 5
    error.age_MAE = (error1.age_MAE + error2.age_MAE + error3.age_MAE + error4.age_MAE + error5.age_MAE) / 5
    error.sex_MAE = (error1.sex_MAE + error2.sex_MAE + error3.sex_MAE + error4.sex_MAE + error5.sex_MAE) / 5

    print("                    Mean Square Error Results: ")
    print("Predicted Ratings by Pearson     MSE: ", error.pearson_MSE)
    print("Predicted Ratings by Kmeans      MSE: ", error.kmeans_MSE)
    print("Predicted Ratings by ACO-Cluster MSE: ", error.ACO_MSE)
    print("Predicted Ratings by ACO-Kmeans  MSE: ", error.ACO_Kmeans_MSE)
    print("Predicted Ratings by Age         MSE: ", error.age_MSE)
    print("Predicted Ratings by Sex         MSE: ", error.sex_MSE)

    print("\n")
    print("                    Root Mean Square Error Results: ")
    print("Predicted Ratings by Pearson     RMSE: ", error.pearson_RMSE)
    print("Predicted Ratings by Kmeans      RMSE: ", error.kmeans_RMSE)
    print("Predicted Ratings by ACO-Cluster RMSE: ", error.ACO_RMSE)
    print("Predicted Ratings by ACO-Kmeans  RMSE: ", error.ACO_Kmeans_RMSE)
    print("Predicted Ratings by Age         RMSE: ", error.age_RMSE)
    print("Predicted Ratings by Sex         RMSE: ", error.sex_RMSE)

    print("\n")
    print("                    Mean Absolute Error Results: ")
    print("Predicted Ratings by Pearson     RMSE: ", error.pearson_MAE)
    print("Predicted Ratings by Kmeans      RMSE: ", error.kmeans_MAE)
    print("Predicted Ratings by ACO-Cluster RMSE: ", error.ACO_MAE)
    print("Predicted Ratings by ACO-Kmeans  RMSE: ", error.ACO_Kmeans_MAE)
    print("Predicted Ratings by Age         RMSE: ", error.age_MAE)
    print("Predicted Ratings by Sex         RMSE: ", error.sex_MAE)

    return error



    error.kmeans_RMSE = ( error1.kmeans_RMSE + error2.kmeans_RMSE + error3.kmeans_RMSE + error4.kmeans_RMSE + error5.kmeans_RMSE) / 5

def create_averages(data):
    # Calculate average value for user ratings and add to user object
    data.user = create_avg_user(data.user, data.n_users, data.user_item_ratings)
    average_ratings_for_age_by_items, average_ratings_for_sex_by_items = create_avg_ratings(data.user, data.n_users,
                                                                                            data.user_item_ratings,
                                                                                            data.n_items)

    return data.user, average_ratings_for_age_by_items, average_ratings_for_sex_by_items


def create_avg_user(user, n_users, utility_clustered):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    for i in range(0, n_users):
        x = utility_clustered[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    return user


def calculate_avg_for_pearson(user_smilarity_top_list, ratings, n_users, n_items):
    smilar_user_average_ratings = np.zeros((n_users, n_items))
    total_rate = 0
    count_rate = 0
    for user_id in range(0, n_users):
        for item_id in range(0, n_items):
            for j in user_smilarity_top_list[user_id]:
                similar_user_id = int(j)
                if ratings[similar_user_id][item_id] != 0:
                    total_rate = total_rate + ratings[similar_user_id][item_id]
                    count_rate = count_rate + 1
            if total_rate != 0 or count_rate != 0:
                total_rate = total_rate / count_rate
            smilar_user_average_ratings[user_id][item_id] = total_rate

    return smilar_user_average_ratings

def calculate_avg_for_aco(user_id, item_id, user, clustered_users, ratings):
    is_break = False
    rate = user[user_id].avg_r
    count = 1
    for cluster in clustered_users:
        if is_break:
            if rate != 0 and count != 0:
                rate = rate / count  # aynı kumedeki kullanicilarin vermis oldugu oyların ortalamasini bul
            break  # outer loop break
        rate = 0
        for j in cluster:  # clusterlar uzerinde gez.
            try:
                if j == user_id:  # verilen kullanicinin hangi clusterda oldugunu bul
                    # similarity.append(cluster)
                    is_break = True
                    # break
                elif ratings[j][item_id] != 0:  # oy kullanmis kullanicilarin oyunu ver
                    rate = rate + ratings[j][item_id]
                    count = count + 1

            except:
                print("An exception occurred", j)

    # print(similarity)
    return rate
def calculate_avg_for_kmeans(ratings, clusters, n_users, n_items):
    average_ratings_for_items = np.zeros((n_items, 5))
    # item sayisi ve 5 cluster.
    for item_id in range(0, n_items):
        avg = KMeansDomain(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for user_id in range(0, n_users):
            cluster_no = clusters[user_id]
            if cluster_no == 0 and ratings[user_id][item_id] != 0:
                avg.avg_0 = avg.avg_0 + ratings[user_id][item_id]
                avg.count_0 = avg.count_0 + 1
            elif cluster_no == 1 and ratings[user_id][item_id] != 0:
                avg.avg_1 = avg.avg_1 + ratings[user_id][item_id]
                avg.count_1 = avg.count_1 + 1
            elif cluster_no == 2 and ratings[user_id][item_id] != 0:
                avg.avg_2 = avg.avg_2 + ratings[user_id][item_id]
                avg.count_2 = avg.count_2 + 1
            elif cluster_no == 3 and ratings[user_id][item_id] != 0:
                avg.avg_3 = avg.avg_3 + ratings[user_id][item_id]
                avg.count_3 = avg.count_3 + 1
            elif cluster_no == 4 and ratings[user_id][item_id] != 0:
                avg.avg_4 = avg.avg_4 + ratings[user_id][item_id]
                avg.count_4 = avg.count_4 + 1

        if avg.count_0 != 0:
            avg.avg_0 = avg.avg_0 / avg.count_0
            average_ratings_for_items[item_id][0] = avg.avg_0
        if avg.count_1 != 0:
            avg.avg_1 = avg.avg_1 / avg.count_1
            average_ratings_for_items[item_id][1] = avg.avg_1
        if avg.count_2 != 0:
            avg.avg_2 = avg.avg_2 / avg.count_2
            average_ratings_for_items[item_id][2] = avg.avg_2
        if avg.count_3 != 0:
            avg.avg_3 = avg.avg_3 / avg.count_3
            average_ratings_for_items[item_id][3] = avg.avg_3
        if avg.count_4 != 0:
            avg.avg_4 = avg.avg_4 / avg.count_4
            average_ratings_for_items[item_id][4] = avg.avg_4

    return average_ratings_for_items


def create_avg_ratings(user, n_users, ratings, n_items):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    average_ratings_for_age_by_items = np.zeros((n_items, 4))
    average_ratings_for_sex_by_items = np.zeros((n_items, 2))
    for item_id in range(0, n_items):  # oge bazli uzerinden geciyoruz
        avg = Average(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for user_id in range(0, n_users):
            if user[user_id].sex == 'M' and ratings[user_id][item_id] != 0:
                avg.avg_male = avg.avg_male + ratings[user_id][item_id]
                avg.count_male = avg.count_male + 1
            elif user[user_id].sex == 'F' and ratings[user_id][item_id] != 0:
                avg.avg_female = avg.avg_female + ratings[user_id][item_id]
                avg.count_female = avg.count_female + 1

            if user[user_id].age < 30 and ratings[user_id][item_id] != 0:
                avg.avg_twenty = avg.avg_twenty + ratings[user_id][item_id]
                avg.count_twenty = avg.count_twenty + 1
            elif user[user_id].age < 40 and ratings[user_id][item_id] != 0:
                avg.avg_thirty = avg.avg_thirty + ratings[user_id][item_id]
                avg.count_thirty = avg.count_thirty + 1
            elif user[user_id].age < 50 and ratings[user_id][item_id] != 0:
                avg.avg_forty = avg.avg_forty + ratings[user_id][item_id]
                avg.count_forty = avg.count_forty + 1
            elif ratings[user_id][item_id] != 0:
                avg.avg_fifty = avg.avg_fifty + ratings[user_id][item_id]
                avg.count_fifty = avg.count_fifty + 1

        if avg.count_male != 0:
            avg.avg_male = avg.avg_male / avg.count_male
            average_ratings_for_sex_by_items[item_id][0] = avg.avg_male
        if avg.count_female != 0:
            avg.avg_female = avg.avg_female / avg.count_female
            average_ratings_for_sex_by_items[item_id][1] = avg.avg_female

        if avg.count_twenty != 0:
            avg.avg_twenty = avg.avg_twenty / avg.count_twenty
            average_ratings_for_age_by_items[item_id][0] = avg.avg_twenty
        if avg.count_thirty != 0:
            avg.avg_thirty = avg.avg_thirty / avg.count_thirty
            average_ratings_for_age_by_items[item_id][1] = avg.avg_thirty
        if avg.count_forty != 0:
            avg.avg_forty = avg.avg_forty / avg.count_forty
            average_ratings_for_age_by_items[item_id][2] = avg.avg_forty
        if avg.avg_fifty != 0:
            avg.avg_fifty = avg.avg_fifty / avg.count_fifty
            average_ratings_for_age_by_items[item_id][3] = avg.avg_fifty

    return average_ratings_for_age_by_items, average_ratings_for_sex_by_items
