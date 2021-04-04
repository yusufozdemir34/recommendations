from domain.Average import Average
from domain.KMeansDomain import KMeansDomain
import numpy as np


def create_averages(data):
    # Calculate average value for user ratings and add to user object
    data.user = create_avg_user(data.user, data.n_users, data.user_item_ratings)
    average_ratings_for_age_by_items, average_ratings_for_sex_by_items = create_avg_ratings(data.user, data.n_users, data.user_item_ratings, data.n_items)

    return data.user, average_ratings_for_age_by_items, average_ratings_for_sex_by_items


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

    return avg, average_ratings_for_items


def create_avg_ratings(user, n_users, ratings, n_items):
    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    average_ratings_for_age_by_items = np.zeros((n_items, 4))
    average_ratings_for_sex_by_items = np.zeros((n_items, 2))
    for item_id in range(0, 1682):  # oge bazli uzerinden geciyoruz
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
