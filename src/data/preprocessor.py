
import re
import pandas as pd
import numpy as np


def normalize_isbn(value):
    value = str(value)
    value = re.sub(r"[^0-9Xx]", "", value)
    return value.upper()


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.min_user = config.get("data.min_user_interactions", 5)
        self.min_item = config.get("data.min_item_interactions", 5)

    def preprocess_bookcrossing(self, raw):
        books = raw["books"].rename(columns={
            "ISBN": "isbn",
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
        })
        users = raw["users"].rename(columns={
            "User-ID": "user_id",
            "Location": "location",
            "Age": "age",
        })
        ratings = raw["ratings"].rename(columns={
            "User-ID": "user_id",
            "ISBN": "isbn",
            "Book-Rating": "rating",
        })

        books["isbn"] = books["isbn"].fillna("").astype(str)
        books["title"] = books["title"].fillna("").astype(str)
        books["author"] = books["author"].fillna("").astype(str)
        books["publisher"] = books["publisher"].fillna("").astype(str)

        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["isbn"] = ratings["isbn"].astype(str)
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings = ratings.dropna(subset=["user_id", "isbn", "rating"])

        books["isbn_norm"] = books["isbn"].apply(normalize_isbn)
        ratings["isbn_norm"] = ratings["isbn"].apply(normalize_isbn)

        books["title_norm"] = books["title"].apply(normalize_text)
        books["author_norm"] = books["author"].apply(normalize_text)

        ratings["interaction"] = 1
        ratings["is_explicit"] = (ratings["rating"] > 0).astype(int)
        ratings["explicit_rating"] = np.where(ratings["rating"] > 0, ratings["rating"], np.nan)

        books = books.drop_duplicates("isbn_norm")

        return {
            "books": books,
            "users": users,
            "ratings": ratings,
            "source": "bookcrossing",
        }

    def preprocess_goodbooks(self, raw):
        books = raw["books"].rename(columns={
            "book_id": "goodbook_id",
            "authors": "author",
            "title": "title",
            "isbn": "isbn",
            "average_rating": "average_rating",
            "ratings_count": "ratings_count",
            "original_publication_year": "year",
            "language_code": "language",
        })

        books["isbn"] = books["isbn"].fillna("").astype(str)
        books["isbn_norm"] = books["isbn"].apply(normalize_isbn)
        books["title"] = books["title"].fillna("").astype(str)
        books["author"] = books["author"].fillna("").astype(str)
        books["title_norm"] = books["title"].apply(normalize_text)
        books["author_norm"] = books["author"].apply(normalize_text)

        book_tags = raw["book_tags"].merge(raw["tags"], on="tag_id", how="left")

        book_tag_text = (
            book_tags
            .sort_values(["goodreads_book_id", "count"], ascending=[True, False])
            .groupby("goodreads_book_id")["tag_name"]
            .apply(lambda x: " ".join(x.dropna().astype(str).head(30)))
            .reset_index()
        )
        book_tag_text.columns = ["goodreads_book_id", "tags"]

        books = books.merge(
            book_tag_text,
            left_on="goodbook_id",
            right_on="goodreads_book_id",
            how="left",
        )
        books["tags"] = books["tags"].fillna("").astype(str)

        ratings = raw["ratings"].copy()
        ratings = ratings.rename(columns={"user_id": "user_id", "book_id": "goodbook_id", "rating": "rating"})
        ratings["user_id"] = "gb_" + ratings["user_id"].astype(str)

        return {
            "books": books,
            "ratings": ratings,
            "source": "goodbooks",
        }

    def preprocess_amazon(self, raw):
        ratings = raw["ratings"].dropna(subset=["Title", "User_id", "review/score"]).copy()
        ratings["review/text"] = ratings["review/text"].fillna("").astype(str)
        ratings["review/summary"] = ratings["review/summary"].fillna("").astype(str)
        ratings["review/score"] = pd.to_numeric(ratings["review/score"], errors="coerce")
        ratings = ratings.dropna(subset=["review/score"])

        ratings["review_text"] = (ratings["review/summary"] + " " + ratings["review/text"]).str.strip()
        ratings["is_positive"] = (ratings["review/score"] >= 4).astype(int)
        ratings["user_id"] = "amazon_" + ratings["User_id"].astype(str)
        ratings["item_id"] = "amazon_" + ratings["Title"].astype(str)

        books = raw["books"].copy()

        return {
            "books": books,
            "ratings": ratings,
            "source": "amazon",
        }

    def build_unified_items(self, bc_processed, gb_processed):
        books_bc = bc_processed["books"]
        goodbooks = gb_processed["books"]

        isbn_mapping = books_bc[["isbn_norm", "title", "author"]].merge(
            goodbooks[["isbn_norm", "goodbook_id", "tags", "average_rating", "ratings_count", "language"]],
            on="isbn_norm",
            how="inner",
        )
        isbn_mapping = isbn_mapping.drop_duplicates("isbn_norm")

        title_mapping = books_bc[["isbn_norm", "title_norm", "author_norm"]].merge(
            goodbooks[["goodbook_id", "title_norm", "author_norm", "tags", "average_rating", "ratings_count", "language"]],
            on=["title_norm", "author_norm"],
            how="inner",
        )
        title_mapping = title_mapping.drop_duplicates("isbn_norm")

        mapping = pd.concat(
            [
                isbn_mapping[["isbn_norm", "goodbook_id", "tags", "average_rating", "ratings_count", "language"]],
                title_mapping[["isbn_norm", "goodbook_id", "tags", "average_rating", "ratings_count", "language"]],
            ],
            ignore_index=True,
        )
        mapping = mapping.drop_duplicates("isbn_norm")

        items = books_bc.merge(mapping, on="isbn_norm", how="left")
        items["item_id"] = items["isbn_norm"]
        items = items.drop_duplicates("item_id")

        items["content"] = (
            items["title"].fillna("") + " " +
            items["author"].fillna("") + " " +
            items["tags"].fillna("")
        )

        return items

    def build_interactions(self, bc_processed):
        ratings = bc_processed["ratings"].copy()
        interactions = pd.DataFrame({
            "user_id": "bc_" + ratings["user_id"].astype(str),
            "item_id": ratings["isbn_norm"].astype(str),
            "rating": ratings["rating"].astype(float),
            "interaction": 1,
            "is_explicit": ratings["is_explicit"].astype(int),
            "source": "bookcrossing",
        })
        return interactions

    def filter_interactions(self, interactions):
        user_counts = interactions.groupby("user_id").size()
        active_users = user_counts[user_counts >= self.min_user].index
        filtered = interactions[interactions["user_id"].isin(active_users)].copy()

        item_counts = filtered.groupby("item_id").size()
        active_items = item_counts[item_counts >= self.min_item].index
        filtered = filtered[filtered["item_id"].isin(active_items)].copy()

        return filtered
