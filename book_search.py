import pandas as pd
def book_titles_from_ids(books_df, ids):
    titles = []
    books_df["book_id"].isin(ids)
        