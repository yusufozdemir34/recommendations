from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from domain.Prediction import Prediction


def get_predictions(dataDTO, clusters):
    # ratings,user, clusters_by_kmeans, kmeans_avg, average_ratings_for_item_kmeans, cluster_users, avg, clusters_by_pearson,pearson_average_ratings):
    # clusters.clusters_by_aco, clusters.averages_ratings_by_demographics, clusters.clusters_by_pearson,
    # clusters.pearson_average_ratings)
    predictions = Prediction(0, 0, 0, 0, 0, 0)
    # n_users = len(user)
    predictions.predicted_ratings_by_pearson = np.copy(dataDTO.user_item_ratings_for_predict)
    predictions.predicted_ratings_by_kmeans = np.copy(dataDTO.user_item_ratings_for_predict)
    predictions.predicted_ratings_by_aco = np.copy(dataDTO.user_item_ratings_for_predict)
    predictions.predicted_ratings_by_aco_kmeans = np.copy(dataDTO.user_item_ratings_for_predict)
    predictions.predicted_rating_by_age = np.copy(dataDTO.user_item_ratings_for_predict)
    predictions.predicted_rating_by_sex = np.copy(dataDTO.user_item_ratings_for_predict)
    for user_id in range(0, dataDTO.n_users):
        for item_id in range(0, dataDTO.n_items):
            predictions.predicted_ratings_by_pearson[user_id][item_id] = clusters.pearson_average_ratings[user_id][
                item_id]

            predictions.predicted_ratings_by_kmeans[user_id][item_id] = predict_for_kmeans(user_id, item_id,
                                                                                           clusters.average_ratings_for_item_kmeans,
                                                                                           clusters.clusters_by_kmeans,
                                                                                           dataDTO.user)
            predictions.predicted_ratings_by_aco[user_id][item_id] = predict_for_aco(user_id, item_id, dataDTO.user,
                                                                                     clusters.clusters_by_aco,
                                                                                     dataDTO.user_item_ratings_for_predict)
            predictions.predicted_ratings_by_aco_kmeans[user_id][item_id] = predict_for_aco_kmeans(user_id, item_id,
                                                                                               clusters.average_ratings_for_item_aco,
                                                                                               clusters.clusters_by_aco_kmeans,
                                                                                               dataDTO.user)

            predictions.predicted_rating_by_age[user_id][item_id] = predict_by_age(user_id, item_id,
                                                                                   clusters.average_ratings_for_age_by_items,
                                                                                   dataDTO.user)
            predictions.predicted_rating_by_sex[user_id][item_id] = predict_by_sex(user_id, item_id,
                                                                                   clusters.average_ratings_for_sex_by_items,
                                                                                   dataDTO.user)

    print("\rPrediction [User:Rating] = [%d:%d]" % (user_id, item_id))

    return predictions


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item clusterı
# top_n - bu benzerlik hesabı için kullanılacak benzer user sayısı.
def prediction_user_rating(user_id, i_id, top_n, n_users, pcs_matrix, user, utility_clustered, clusternumber):
    similarity = []
    for i in range(0, n_users):
        if i + 1 != user_id:
            similarity.append(pcs_matrix[user_id - 1][i])
    temp = 0  # norm(n_users, utility_clustered, user, clusternumber)
    temp = np.delete(temp, user_id - 1, 0)
    top = [x for (y, x) in sorted(zip(similarity, temp), key=lambda pair: pair[0], reverse=True)]
    # top: benzerlik ve oylama matrislerinin zip ile eşleşmesi sonucu sorted ile sıralanması ile
    # en yüksek benzerlik oranına sahip bireylerin oylarını saklar.
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id - 1] != float(
                'Inf'):  # infinitive : sınırsız bir üst değer işlevi görür. bu işin sonuna kadar yani
            s += top[i][i_id - 1]  # top'daki oyların toplamı
            c += 1  # oy sayısı. bu hem ortalama için hem de oy olup olmadığı kontrolü için
    rate = user[user_id - 1].avg_r if c == 0 else s / float(c) + user[user_id - 1].avg_r
    # eğer hiç oy yoksa kullanıcının kendi ortalama oyunu kabul et
    # oy varsa en benzer kullanıcıların o film için verdiği oyların ortalamasını kullanıcı için ata. USER-BASED
    if rate < 1.0:
        return 1.0
    elif rate > 5.0:
        return 5.0
    else:
        return rate


def prediction_user_rating(X_train, y_train, X_test):
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    # model.fit(features, label)

    # Predict Output
    predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    return y_pred


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item clusterı
# top_n - bu benzerlik hesabı için kullanılacak benzer user sayısı.
def predict(user_id, i_id, top_n, n_users, pcs_matrix, user, clustered_user, clusternumber, temp):
    similarity = []
    for i in range(0, n_users):
        if i + 1 != user_id:
            similarity.append(pcs_matrix[user_id - 1][i])
    # temp = norm(n_users, clustered_user, user, clusternumber)
    # temp = preprocessing.normalize(clustered_user)
    # temp = np.delete(temp, user_id - 1, 0)
    top = [x for (y, x) in sorted(zip(similarity, temp), key=lambda pair: pair[0], reverse=True)]
    # top: benzerlik ve oylama matrislerinin zip ile eşleşmesi sonucu sorted ile sıralanması ile
    # en yüksek benzerlik oranına sahip bireylerin oylarını saklar.
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id - 1] != float(
                'Inf'):  # infinitive : sınırsız bir üst değer işlevi görür. bu işin sonuna kadar yani
            s += top[i][i_id - 1]  # top'daki oyların toplamı
            c += 1  # oy sayısı. bu hem ortalama için hem de oy olup olmadığı kontrolü için
    rate = user[user_id - 1].avg_r if c == 0 else s / float(c) + user[user_id - 1].avg_r
    # eğer hiç oy yoksa kullanıcının kendi ortalama oyunu kabul et
    # oy varsa en benzer kullanıcıların o film için verdiği oyların ortalamasını kullanıcı için ata. USER-BASED
    if rate < 1.0:
        return 1.0
    elif rate > 5.0:
        return 5.0
    else:
        return rate


def predict_by_age(user_id, item_id, average_ratings_for_age_by_items, user):
    rate = 0
    if user[user_id].age < 30:
        rate = average_ratings_for_age_by_items[item_id][0]  # avg.avg_twenty
    elif user[user_id].age < 40:
        rate = average_ratings_for_age_by_items[item_id][1]
    elif user[user_id].age < 50:
        rate = average_ratings_for_age_by_items[item_id][2]
    else:
        rate = average_ratings_for_age_by_items[item_id][3]
    return rate


def predict_by_sex(user_id, item_id, average_ratings_for_sex_by_items, user):
    rate = 0
    if user[user_id].sex == 'M':
        rate = average_ratings_for_sex_by_items[item_id][0]
    else:
        rate = average_ratings_for_sex_by_items[item_id][1]
    return rate


def predict_by_pearson(user_id, pearson_matrix, user):
    return user


def predict_for_aco_kmeans (user_id, item_id, average_ratings_for_item_kmeans, clusters_by_kmeans, user):
    rate = 0

    if clusters_by_kmeans[user_id] == 0:
        if average_ratings_for_item_kmeans[item_id][0] != 0:
            rate = average_ratings_for_item_kmeans[item_id][0]
        else:
            rate = user[user_id].avg_r
    elif clusters_by_kmeans[user_id] == 1:
        if average_ratings_for_item_kmeans[item_id][1] != 0:
            rate = average_ratings_for_item_kmeans[item_id][1]
        else:
            rate = user[user_id].avg_r
    elif clusters_by_kmeans[user_id] == 2:
        if average_ratings_for_item_kmeans[item_id][2] != 0:
            rate = average_ratings_for_item_kmeans[item_id][2]
        else:
            rate = user[user_id].avg_r
    elif clusters_by_kmeans[user_id] == 3:
        if average_ratings_for_item_kmeans[item_id][3] != 0:
            rate = average_ratings_for_item_kmeans[item_id][3]
        else:
            rate = user[user_id].avg_r
    elif clusters_by_kmeans[user_id] == 4:
        if average_ratings_for_item_kmeans[item_id][4] != 0:
            rate = average_ratings_for_item_kmeans[item_id][4]
        else:
            rate = user[user_id].avg_r

    return rate


def predict_for_kmeans (user_id, item_id, average_ratings_for_item_kmeans, clusters_by_kmeans, user):
    rate = 0

    if clusters_by_kmeans[user_id] == 0:
        if average_ratings_for_item_kmeans[item_id][0] != 0:
            rate = (average_ratings_for_item_kmeans[item_id][0] + user[user_id].avg_r) / 2
    elif clusters_by_kmeans[user_id] == 1:
        if average_ratings_for_item_kmeans[item_id][1] != 0:
            rate = (average_ratings_for_item_kmeans[item_id][1] + user[user_id].avg_r) / 2
    elif clusters_by_kmeans[user_id] == 2:
        if average_ratings_for_item_kmeans[item_id][2] != 0:
            rate = (average_ratings_for_item_kmeans[item_id][2] + user[user_id].avg_r) / 2
    elif clusters_by_kmeans[user_id] == 3:
        if average_ratings_for_item_kmeans[item_id][3] != 0:
            rate = (average_ratings_for_item_kmeans[item_id][3] + user[user_id].avg_r) / 2
    elif clusters_by_kmeans[user_id] == 4:
        if average_ratings_for_item_kmeans[item_id][4] != 0:
            rate = ( average_ratings_for_item_kmeans[item_id][4] + user[user_id].avg_r) / 2

    return rate


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item
# bu yöntemle bir kullanicin oy vermediği öğeye verebileceği olası oyu tahmin ediyoruz
# once kullanicinin bulunduğu kümeyi bulmaliyiz.
# bu kümede verilen öğe için verilmiş ortalama oyu bulup kullanicinin kendi ortalamasına eklemeliyiz.
def predict_for_aco(user_id, item_id, user, clustered_users, ratings):
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
