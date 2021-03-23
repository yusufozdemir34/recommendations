from domain.Average import Average
from domain.KMeansDomain import KMeansDomain
import numpy as np


def create_averages(user, n_users, ratings, n_items):
    # Calculate average value for user ratings and add to user object
    user = create_avg_user(user, n_users, ratings)
    avg = create_avg_ratings(user, n_users, ratings, n_items)

    return user, avg


def create_avg_user(user, n_users, utility_clustered):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    for i in range(0, n_users):
        x = utility_clustered[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    return user


def calculate_avg_for_pearson(user_smilarity_top_list, ratings, n_users, n_items):
    smilar_user_average_ratings = np.zeros((943, 1682))
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

    return avg ,average_ratings_for_items


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
