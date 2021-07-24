import pandas as pd
def book_titles_from_ids(books_df, ids):
    return list(books_df[books_df["book_id"].isin(id)]["title"])

def user_ratings_by_id(ratings_df, user_id):
    return 