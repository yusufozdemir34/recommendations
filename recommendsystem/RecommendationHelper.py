import re
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# Pearson Korelasyonu. Userlar arasında dolayısı ile user based.
# item based olması için itemler arasında ilişki hesabı da yapılacak.
def pearson(x, y, utility_clustered, user):
    num = 0
    den1 = 0
    den2 = 0
    A = utility_clustered[x - 1]
    B = utility_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den


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


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item
# bu yöntemle bir kullanicin oy vermediği öğeye verebileceği olası oyu tahmin ediyoruz
# once kullanicinin bulunduğu kümeyi bulmaliyiz.
# bu kümede verilen öğe için verilmiş ortalama oyu bulup kullanicinin kendi ortalamasına eklemeliyiz.
def predict1(user_id, i_id, top_n, n_users, pcs_matrix, user, clustered_user, clusternumber, temp):
    similarity = []
    is_break = False
    rate = user[user_id].avg_r
    count = 1
    for cluster in clustered_user:
        if is_break:
            if rate != 0 and count != 0:
                rate = rate / count  # aynı kumedeki kullanicilarin vermis oldugu oyların ortalamasini bul
            break  # outer loop break
        rate = 0
        for j in cluster:  # clusterlar uzerinde gez.
            try:
                if j == user_id:  # verilen kullanicinin hangi clusterda oldugunu bul
                    similarity.append(cluster)
                    is_break = True
                    # break
                elif temp[j][i_id] != 0:  # oy kullanmis kullanicilarin oyunu ver
                    rate = rate + temp[j][i_id]
                    count = count + 1

            except:
                print("An exception occurred", j)

    # print(similarity)
    return rate


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item
# bu yöntemle bir kullanicin oy vermediği öğeye verebileceği olası oyu tahmin ediyoruz
# once kullanicinin bulunduğu kümeyi bulmaliyiz.
# bu kümede verilen öğe için verilmiş ortalama oyu bulup kullanicinin kendi ortalamasına eklemeliyiz.
def predict2(user_id, i_id, top_n, n_users, pcs_matrix, user, clustered_user, clusternumber, temp):
    similarity = []
    is_break = False
    rate = user[user_id].avg_r
    count = 1
    for cluster in clustered_user:
        if is_break:
            if rate != 0 and count != 0:
                rate = rate / count  # aynı kumedeki kullanicilarin vermis oldugu oyların ortalamasini bul
            break  # outer loop break
        rate = 0
        for j in cluster:  # clusterlar uzerinde gez.
            try:
                if j == user_id:  # verilen kullanicinin hangi clusterda oldugunu bul
                    similarity.append(cluster)
                    is_break = True
                    # break
                elif temp[j][i_id] != 0:  # oy kullanmis kullanicilarin oyunu ver
                    rate = rate + temp[j][i_id]
                    count = count + 1

            except:
                print("An exception occurred", j)

    # print(similarity)
    return rate


def norm(n_users, clustered_user, user, n_cluster):
    normalize = np.zeros((n_users, n_cluster))
    for i in range(0, n_cluster):
        j = 0
        try:
            for cluster in clustered_user[i]:
                if cluster != 0:
                    normalize[i][j] = cluster - user[i].avg_r
                else:
                    normalize[i][j] = float('Inf')
                j = j + 1
        except:
            print("An exception occurred")

    return normalize


def set_one_for_max_avg_value_others_zero(data):
    max_matrix = data.max(0)
    avg_matrix = np.mean(max_matrix)
    for_i_size = np.size(data, 1)
    for_y_size = np.size(data, 0)

    # delta_math ın içindeki en yüksek avg yi bul. sonra en yüksek avg ye bir de. diğerleri sıfırdır.
    for j in range(0, for_i_size):
        for i in range(0, for_y_size):
            if data[i][j] == max_matrix[j] and max_matrix[j] > avg_matrix * 0.92:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


def isaverage(delta_mat):
    avg = delta_mat.mean()
    for i in range(0, np.size(delta_mat, 0)):
        for j in range(0, np.size(delta_mat, 1)):
            if delta_mat[i][j] > avg / 1000:
                delta_mat[i][j] = 1
            else:
                delta_mat[i][j] = 0
    return delta_mat


def cluster_means(utility, clusters):
    cluster_avg = []
    # calculate average of each line (user)

    for i in range(0, len(clusters) - 1):
        temp = []
        for cluster in clusters[i]:
            temp.append(utility[cluster])
        cluster_avg.append(np.mean(temp))

        # for j in range(0, len(clusters[i]) - 1):
        #   temp.append(utility[clusters[i][j]])
        # cluster_avg.append(np.mean(temp))

    return cluster_avg


def cluster_mean_from_components(utility, components):
    cluster_avg = []
    for i in range(0, len(utility)):
        cluster_avg.append(np.mean(utility[i]))

    return cluster_avg
    #
    # cluster_avg = []
    # # calculate average of each line (user)
    # for nodes in components:
    #     temp = []
    #     for node in nodes:
    #         temp.append(utility[node])
    #     cluster_avg.append(np.mean(temp))
    # return cluster_avg


def get_prediction(utility, pcs_matrix, user, cluster_users):
    n_users = len(user)
    n_cluster = len(cluster_users)
    utility_copy = np.copy(utility)
    temp_normalized = norm(n_users, cluster_users, user, n_cluster)
    for i in range(0, n_users):
        for j in range(0, n_cluster):
            if utility_copy[i][j] == 0:  # oy verilmemis item lara oy tahmini yap
                utility_copy[i][j] = predict1(i, j, 2, n_users, pcs_matrix, user, cluster_users, n_cluster,
                                              utility)
    print("\rPrediction [User:Rating] = [%d:%d]" % (i, j))

    return utility_copy


def mean_square_error(test, utility, n_users, n_items):
    # test datası ile tehmin arasında MSE
    y_true = []
    y_pred = []
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                y_true.append(test[i][j])
                y_pred.append(utility[i][j])

    print("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))


def createCluster(result):
    clusterNumber = result.max()
    clusterUser = []
    for m in range(0, clusterNumber):
        clusterUser.append(set())
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            try:
                clusterUser[result[i][j]].add(i)
            except:
                print("An exception occurred", i, j, result[i][j])
    return clusterUser


def create_avg_user(user, n_users, utility_clustered):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    for i in range(0, n_users):
        x = utility_clustered[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    return user


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item clusterı
# top_n - bu benzerlik hesabı için kullanılacak benzer user sayısı.
def prediction_user_rating(user_id, i_id, top_n, n_users, pcs_matrix, user, utility_clustered, clusternumber):
    similarity = []
    for i in range(0, n_users):
        if i + 1 != user_id:
            similarity.append(pcs_matrix[user_id - 1][i])
    temp = norm(n_users, utility_clustered, user, clusternumber)
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


def choose_maxvalue_from_relation(result):
    result = set_one_for_max_avg_value_others_zero(result)
    return result
