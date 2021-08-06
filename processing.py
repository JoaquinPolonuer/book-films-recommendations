import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Books:
    def __init__(self, global_path):
        self.objects = pd.read_csv(global_path+ "/books.csv")
        
class Tags:
    def __init__(self, global_path):
        self.objects = pd.read_csv(global_path+ "/tags.csv")

class BookTags:
    def __init__(self, global_path):
        self.objects = pd.read_csv(global_path+"/book_tags.csv")
        self._books = Books(global_path)
        self._tags = Tags(global_path)
        self.objects = pd.merge(self.objects, self._books.objects[["goodreads_book_id","book_id"]], on="goodreads_book_id", how="right")

        book_ids = self.objects.book_id.unique().shape[0]
        tag_ids = self.objects.tag_id.unique().shape[0]
        
        self.matrix = np.zeros((book_ids, tag_ids))
        for row in self.objects.itertuples():
            self.matrix[row.book_id-1, row.tag_id-1] = 1
    
    def get_book_similarities(self, book):
        book_id = book.book_id
        book_row = self.matrix[book_id-1]
        similarities_to_book = cosine_similarity([book_row],self.matrix)[0]
        return similarities_to_book

if __name__ == "__main__":
    goodbooks_dataset_path = "/home/joacopolo/Documents/software/python/draft-projects/books-recommender/goodbooks-10k"
    books = Books(goodbooks_dataset_path)
    tags = Tags(goodbooks_dataset_path)
    book_tags = BookTags(goodbooks_dataset_path)

    book = books.objects.iloc[2]
    book_similarities = book_tags.get_book_similarities(book)
    print("Nuestro libro es", book["title"], "los mas parecidos son")
    [print(books.objects.iloc[i]["title"], "("+str(int(book_similarities[i]*100))+"%)") for i in book_similarities.argsort()[::-1][:10]]