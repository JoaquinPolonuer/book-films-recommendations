import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BookDataset():
    def __init__(self, 
                 books,
                 ratings,
                 ratings_matrix,
                ):
        self.ratings = ratings
        self.books = books
        self.ratings_matrix = ratings_matrix
        self.ratings_dict = {}
        self.books["title_lowercase"] = books["title"].str.lower()

    def get_book_by_title(self, title):
        match = self.books[self.books["title_lowercase"].str.contains(title.lower())]
        return match.head(1)

    def get_book_by_id(self, book_id):
        return self.books[self.books["book_id"]==book_id]

    
    def rate_book(self, book_id, rating):
        self.ratings_dict[book_id] = rating
        return self.ratings_dict

    def see_ratings_dict(self, ratings_dict = None):
        if ratings_dict is None:
            ratings_dict = self.ratings_dict
        ratings_dict_title = {}
        for book_id, rating in ratings_dict.items():
            title = self.get_book_by_id(book_id)["title"]
            ratings_dict_title[str(title)] = rating
        return ratings_dict_title

    def generate_ratings_row(self):
        ratings_row = np.zeros((n_books))
        for book_id, rating in self.ratings_dict.items():
            ratings_row[book_id-1] = rating
        return ratings_row

    def get_most_similar_user_id(self,my_row):
        similarities = []
        for row in self.ratings_matrix:
            similarities.append(cosine_similarity([row],[my_row]))
        most_similar_user = np.argmax(similarities)
        similarity = similarities[most_similar_user]
        return most_similar_user, similarity

    def ratings_dict_from_row(self, rating_row):
        ratings_dict = {}
        for i, rating in enumerate(rating_row):
            if rating != 0:
                ratings_dict[i+1] = rating
        return ratings_dict
