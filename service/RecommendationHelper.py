import re
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from scipy.stats import pearsonr

from domain.Average import Average
from domain.Prediction import Prediction


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


def create_avg_ratings(user, n_users, ratings, n_items):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    avg = Average(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # for j in range(0, n_users):
    #     if user[j].sex == 'M':  # if kontrolunu bir kere yapmak gerekir. su an her seferinde yapiliyor
    #         avg.count_male = avg.count_male + 1
    #     else:
    #         avg.count_female = avg.count_female + 1
    #     if user[j].age < 30:  # if kontrolunu bir kere yapmak gerekir. su an her seferinde yapiliyor
    #         avg.count_twenty = avg.count_twenty + 1
    #     elif user[j].age < 40:
    #         avg.count_thirty = avg.count_thirty + 1
    #     elif user[j].age < 50:
    #         avg.count_forty = avg.count_forty + 1
    #     else:
    #         avg.count_fifty = avg.count_fifty + 1
    #     for i in range(0, 1682):  # oge bazli uzerinden geciyoruz
    #         if user[j].sex == 'M':
    #             avg.avg_male = avg.avg_male + ratings[j][i]

    for i in range(0, 1682):  # oge bazli uzerinden geciyoruz
        for j in range(0, n_users):
            if user[j].sex == 'M' and ratings[j][i] != 0:
                avg.avg_male = avg.avg_male + ratings[j][i]
                avg.count_male = avg.count_male + 1
            elif user[j].sex == 'F' and ratings[j][i] != 0:
                avg.avg_female = avg.avg_female + ratings[j][i]
                avg.count_female = avg.count_female + 1

            if user[j].age < 30 and ratings[j][i] != 0:
                avg.avg_twenty = avg.avg_twenty + ratings[j][i]
                avg.count_twenty = avg.count_twenty + 1
            elif user[j].age < 40 and ratings[j][i] != 0:
                avg.avg_thirty = avg.avg_thirty + ratings[j][i]
                avg.count_thirty = avg.count_thirty + 1
            elif user[j].age < 50 and ratings[j][i] != 0:
                avg.avg_forty = avg.avg_forty + ratings[j][i]
                avg.count_forty = avg.count_forty + 1
            elif ratings[j][i] != 0:
                avg.avg_fifty = avg.avg_fifty + ratings[j][i]
                avg.count_fifty = avg.count_fifty + 1

    avg.avg_twenty = avg.avg_twenty / avg.count_twenty
    avg.avg_thirty = avg.avg_thirty / avg.count_thirty
    avg.avg_forty = avg.avg_forty / avg.count_forty
    avg.avg_fifty = avg.avg_fifty / avg.count_fifty

    avg.avg_male = avg.avg_male / avg.count_male
    avg.avg_female = avg.avg_female / avg.count_female

    return avg


def create_avg_user(user, n_users, utility_clustered):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    for i in range(0, n_users):
        x = utility_clustered[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    return user



def choose_maxvalue_from_relation(result):
    result = set_one_for_max_avg_value_others_zero(result)
    return result
