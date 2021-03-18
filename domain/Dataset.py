import re
# User - Item ve Rating verilerini belirlenecek dizilere eklemeyi sağlayacak.
from domain.Item import Item
from domain.Rating import Rating
from domain.User import User


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
