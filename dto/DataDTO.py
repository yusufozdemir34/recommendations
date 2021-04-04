from domain.Average import Average


class DataDTO:
    def __init__(self, user_item_ratings_for_predict, user_item_ratings, user, item, user_user_pearson, n_users, n_items):
        self.user_item_ratings_for_predict = user_item_ratings_for_predict
        self.user_item_ratings = user_item_ratings
        self.user = user
        self.item = item
        self.user_user_pearson = user_user_pearson
        self.n_users = n_users
        self.n_items = n_items
        self.avg = Average(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)