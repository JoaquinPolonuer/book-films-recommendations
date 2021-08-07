import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
class Books:
    def __init__(self, global_path):
        self.objects = pd.read_csv(global_path+ "/books.csv")
    
    def get_book_id_by_title(self, title):
        return self.objects[self.objects.title.str.contains(title, flags=re.IGNORECASE)]
    
    def get_books(self):
        return list(self.objects.T.to_dict().values())

    def get_book_by_id(self, book_id):
        return self.objects.iloc[book_id-1]     
           
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
    goodbooks_dataset_path = "dataset/"
    books = Books(goodbooks_dataset_path)
    tags = Tags(goodbooks_dataset_path)
    book_tags = BookTags(goodbooks_dataset_path)

    while True:
        title = input("Title: ")
        print("")
        matches = books.get_book_id_by_title(title)
        [print(match.title, match.book_id) for match in matches.itertuples()]
        print("")
        if input("Encontraste tu libro? (y/n)") == "y":
            print("")
            break

    book_id = int(input("Ingresa el id del libro:"))
    print("")
    book = books.get_book_by_id(book_id)
    book_similarities = book_tags.get_book_similarities(book)
    print("Nuestro libro es", book["title"],"...")
    print("Los mas parecidos son:\n")
    [print(books.objects.iloc[i]["title"], "-", books.objects.iloc[i]["authors"], "("+str(int(book_similarities[i]*100))+"%)") for i in book_similarities.argsort()[::-1][1:10]]