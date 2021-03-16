import re
import numpy as np
from scipy.stats import pearsonr


class User:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0


class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
                 unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
                 drama, fantasy, film_noir, horror, musical, mystery, romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)


class Rating:
    def __init__(self, user_id, item_id, rating, time):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.time = time


class Average:
    def __init__(self, avg_male, avg_female, avg_twenty, avg_thirty, avg_forty, avg_fifty, count_male, count_female,
                 count_twenty, count_thirty,
                 count_forty, count_fifty):
        self.avg_male = int(avg_male)
        self.avg_female = int(avg_female)
        self.avg_twenty = int(avg_twenty)
        self.avg_thirty = int(avg_thirty)
        self.avg_forty = int(avg_forty)
        self.avg_fifty = int(avg_fifty)

        self.count_male = int(count_male)
        self.count_female = int(count_female)
        self.count_twenty = int(count_twenty)
        self.count_thirty = int(count_thirty)
        self.count_forty = int(count_forty)
        self.count_fifty = int(count_fifty)


# User - Item ve Rating verilerini belirlenecek dizilere eklemeyi sağlayacak.
class Dataset:
    def load_users(self, file, u):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 5)
            if len(e) == 5:
                u.append(User(e[0], e[1], e[2], e[3], e[4]))
        f.close()

    def load_items(self, file, i):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 24)
            if len(e) == 24:
                i.append(Item(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], \
                              e[11], e[12], e[13], e[14], e[15], e[16], e[17], e[18], e[19], e[20], e[21], \
                              e[22], e[23]))
        f.close()

    def load_ratings(self, file, r):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                r.append(Rating(e[0], e[1], e[2], e[3]))
        f.close()


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
    utility = np.zeros((n_users, n_items))
    for r in rating:
        utility[r.user_id - 1][r.item_id - 1] = r.rating

    # print(utility)

    test = np.zeros((n_users, n_items))
    for r in rating_test:
        test[r.user_id - 1][r.item_id - 1] = r.rating
    # verinin bir kismini sildim. sildigim datalari -1 olarak isaretliyoruz.
    # daha sonra bunlari tahmin edeceğimizde hangilerini tahmin ettiğimizi bilmek icin -1 yapiyoruz.
    for i in range(0, n_users):
        for j in range(0, n_items):
            if utility[i][j] != test[i][j]:
                utility[i][j] == -1

    # clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
    # prediction daki [cluster.labels_[j] yerine ne ekleycegiz
    pcs_matrix = np.zeros((n_users, n_users))

    for i in range(0, n_users):
        for j in range(0, i):
            if i != j:
                A = utility[i]
                B = utility[j]
                pcs_matrix[i][j], _ = pearsonr(A, B)
    # store all records to nm array by being binary
    save_model_as_np(utility, test, user, item, pcs_matrix)
    return utility, test, user, item, pcs_matrix


def save_model_as_np(utility, test, user, item, pcs_matrix):
    np.save("../data/runned_data/np_user", user)
    np.save("../data/runned_data/np_item", item)
    np.save("../data/runned_data/np_test", test)
    np.save("../data/runned_data/np_pcs_matrix", pcs_matrix)
    np.save("../data/runned_data/np_utility", utility)


def load_model_as_np():
    user = np.load("../data/runned_data/np_user.npy", allow_pickle=True)
    item = np.load("../data/runned_data/np_item.npy", allow_pickle=True)
    test = np.load("../data/runned_data/np_test.npy", allow_pickle=True)
    pcs_matrix = np.load("../data/runned_data/np_pcs_matrix.npy", allow_pickle=True)
    utility = np.load("../data/runned_data/np_utility.npy", allow_pickle=True)

    return utility, test, user, item, pcs_matrix
