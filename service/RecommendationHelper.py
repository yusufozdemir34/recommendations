import numpy as np
from domain.Average import Average


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


