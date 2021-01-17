import re
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
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
def predict(user_id, i_id, top_n, n_users, pcs_matrix, user, clustered_user, clusternumber):
    similarity = []
    for i in range(0, n_users):
        if i + 1 != user_id:
            similarity.append(pcs_matrix[user_id - 1][i])
    temp = norm(n_users, clustered_user, user, clusternumber)
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


def norm(n_users, clustered_user, user, n_cluster):
    normalize = np.zeros((n_users, n_users))
    for i in range(0, n_users):
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


def set_one_for_max_avg_value_others_zero(delta_mat):
    max_matrix = delta_mat.max(0)
    for_i_size = np.size(delta_mat, 1) - 1
    for_y_size = np.size(delta_mat, 0)

    # delta_math ın içindeki en yüksek avg yi bul. sonra en yüksek avg ye bir de. diğerleri sıfırdır.
    for j in range(0, for_i_size):
        for i in range(0, for_y_size):
            if delta_mat[i][j] == max_matrix[j]:
                delta_mat[i][j] = 1
            else:
                delta_mat[i][j] = 0
    return delta_mat


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


def get_prediction(utility, pcs_matrix, user, cluster_users):
    cluster_users.pop(0)
    maxim_cluster = max(cluster_users)
    n_users = len(user)
    n_cluster = len(maxim_cluster)  # 19
    utility_copy = np.copy(utility)
    for i in range(0, n_users):
        for j in range(0, n_cluster):
            if utility_copy[i][j] == 0:
                utility_copy[i][j] = predict(i + 1, j + 1, 2, n_users, pcs_matrix, user, cluster_users, n_cluster)
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
    d.load_ratings("../data/ua.base", rating)
    d.load_ratings("../data/ua.test", rating_test)

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

    # clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
    # prediction daki [cluster.labels_[j] yerine ne ekleycegiz
    pcs_matrix = np.zeros((n_users, n_users))

    for i in range(0, n_users):
        for j in range(0, i):
            if i != j:
                A = utility[i]
                B = utility[j]
                pcs_matrix[i][j], _ = pearsonr(A, B)

    return user, item, test, pcs_matrix, utility, n_users, n_items
