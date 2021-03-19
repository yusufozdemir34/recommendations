import numpy as np
from scipy.stats import pearsonr

from domain.Dataset import Dataset
from domain.User import User


def prepare_data():
    # find user_user_pearson relative to pearson algorithm
    user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson = load_model_as_np()
    n_users = len(user)
    n_items = len(item)
    for i in range(0, n_users):
        for j in range(0, n_items):
            if user_item_ratings_for_predict[i][j] != user_item_ratings[i][j]:
                user_item_ratings_for_predict[i][j] = -1
                print("-1 yapilcaklar i: ", i, " j: ", j)

    return user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items


def create_model():
    # verilerin tutulacağı diziler
    user = []
    item = []

    rating = []
    rating_test = []

    # Dataset class kullanarak veriyi dizilere aktarma
    d = Dataset()
    d.load_users("../data/u.user", user)
    d.load_items("../data/u.item", item)
    d.load_ratings("../data/uayusuf.base", rating)
    d.load_ratings("../data/ua.base", rating_test)

    n_users = len(user)
    n_items = len(item)
    n_users

    # utility user-item tablo sonucu olarak rating tutmaktadır.
    # NumPy sıfırlar işlevi, yalnızca sıfır içeren NumPy dizileri oluşturmanıza olanak sağlar.
    # Daha da önemlisi, bu işlev dizinin tam boyutlarını belirlemenizi sağlar.
    # Ayrıca tam veri türünü belirlemenize de olanak tanır.
    user_item_ratings = np.zeros((n_users, n_items))
    for r in rating:
        user_item_ratings[r.user_id - 1][r.item_id - 1] = r.rating

    # print(utility)

    base_user_item_ratings = np.zeros((n_users, n_items))
    for r in rating_test:
        base_user_item_ratings[r.user_id - 1][r.item_id - 1] = r.rating

    # clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
    # prediction daki [cluster.labels_[j] yerine ne ekleycegiz
    pcs_matrix = np.zeros((n_users, n_users))

    for i in range(0, n_users):
        for j in range(0, i):
            if i != j:
                A = user_item_ratings[i]
                B = user_item_ratings[j]
                pcs_matrix[i][j], _ = pearsonr(A, B)
    # verinin bir kismini sildim. sildigim datalari -1 olarak isaretliyoruz.
    # daha sonra bunlari tahmin edeceğimizde hangilerini tahmin ettiğimizi bilmek icin -1 yapiyoruz.
    for i in range(0, n_users):
        for j in range(0, n_items):
            if user_item_ratings[i][j] != base_user_item_ratings[i][j]:
                user_item_ratings[i][j] == -1
    # store all records to nm array by being binary
    save_model_as_np(user_item_ratings, base_user_item_ratings, user, item, pcs_matrix)
    return user_item_ratings, base_user_item_ratings, user, item, pcs_matrix


def save_model_as_np(utility, test, user, item, pcs_matrix):
    np.save("../data/runned_data/np_user", user)
    np.save("../data/runned_data/np_item", item)
    np.save("../data/runned_data/np_test", test)
    np.save("../data/runned_data/np_pcs_matrix", pcs_matrix)
    np.save("../data/runned_data/np_utility", utility)


def load_model_as_np():
    # user = User(0,0,0,0,0)
    user = np.load("../data/runned_data/np_user.npy", allow_pickle=True)
    item = np.load("../data/runned_data/np_item.npy", allow_pickle=True)
    test = np.load("../data/runned_data/np_test.npy", allow_pickle=True)
    pcs_matrix = np.load("../data/runned_data/np_pcs_matrix.npy", allow_pickle=True)
    utility = np.load("../data/runned_data/np_utility.npy", allow_pickle=True)

    return utility, test, user, item, pcs_matrix
