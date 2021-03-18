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
