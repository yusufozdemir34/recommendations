import numpy as np
from scipy.stats import pearsonr

from domain.Dataset import Dataset
from domain.User import User
from dto.DataDTO import DataDTO



def prepare_data():
    # find user_user_pearson relative to pearson algorithm
    user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson = load_model_as_np()
    n_users = len(user)
    n_items = len(item)
    for i in range(0, n_users):
        for j in range(0, n_items):
            if user_item_ratings_for_predict[i][j] != user_item_ratings[i][j]:
                user_item_ratings_for_predict[i][j] = -1
                # print("-1 yapilarak tahmin edilecek oylar i: ", i, " j: ", j)

    dataDTO = DataDTO(user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items)
    return dataDTO
    # user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items


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
    d.load_ratings("../data/u3.base", rating)
    d.load_ratings("../data/u3.test", rating_test)

    n_users = len(user)
    n_items = len(item)
    n_users

    # utility user-item tablo sonucu olarak rating tutmaktadır.
    # NumPy sıfırlar işlevi, yalnızca sıfır içeren NumPy dizileri oluşturmanıza olanak sağlar.
    # Daha da önemlisi, bu işlev dizinin tam boyutlarını belirlemenizi sağlar.
    # Ayrıca tam veri türünü belirlemenize de olanak tanır.
    training_user_item_ratings = np.zeros((n_users, n_items))
    for r in rating:
        training_user_item_ratings[r.user_id - 1][r.item_id - 1] = r.rating

    # print(utility)

    test_user_item_ratings = np.zeros((n_users, n_items))
    for r in rating_test:
        test_user_item_ratings[r.user_id - 1][r.item_id - 1] = r.rating

    # clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
    # prediction daki [cluster.labels_[j] yerine ne ekleycegiz
    pcs_matrix = np.zeros((np.size(training_user_item_ratings, 0), np.size(training_user_item_ratings, 0)))

    for i in range(0, np.size(training_user_item_ratings, 0)):
        for j in range(0, i):
            if i != j:
                A = training_user_item_ratings[i]
                B = training_user_item_ratings[j]
                pcs_matrix[i][j], _ = pearsonr(A, B)

    n_users = np.size(test_user_item_ratings, 0)
    n_items = np.size(test_user_item_ratings, 1)

    dataDTO = DataDTO(test_user_item_ratings, training_user_item_ratings, user, item, pcs_matrix, n_users,
                      n_items)

    return dataDTO


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
