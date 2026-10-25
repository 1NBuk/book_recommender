
import os
import re
import glob
import pandas as pd
import numpy as np


def find_file(root, name):
    matches = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not matches:
        raise FileNotFoundError(f"File not found: {name} in {root}")
    return matches[0]


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = config.get("data.raw_dir", "data/raw")

    def load(self, source):
        if source == "bookcrossing":
            return self._load_bookcrossing()
        if source == "goodbooks":
            return self._load_goodbooks()
        if source == "amazon":
            return self._load_amazon()
        raise ValueError(f"Unknown source: {source}")

    def load_all(self):
        sources = self.config.get("data.sources", [])
        result = {}
        for source in sources:
            result[source] = self.load(source)
        return result

    def _load_bookcrossing(self):
        root = os.path.join(self.raw_dir, "bookcrossing")
        books_path = find_file(root, "Books.csv")
        users_path = find_file(root, "Users.csv")
        ratings_path = find_file(root, "Ratings.csv")

        books = pd.read_csv(books_path, encoding="latin-1", sep=",", on_bad_lines="skip")
        users = pd.read_csv(users_path, encoding="latin-1", sep=",", on_bad_lines="skip")
        ratings = pd.read_csv(ratings_path, encoding="latin-1", sep=",", on_bad_lines="skip")

        return {
            "books": books,
            "users": users,
            "ratings": ratings,
            "source": "bookcrossing",
        }

    def _load_goodbooks(self):
        root = os.path.join(self.raw_dir, "goodbooks")
        books_path = find_file(root, "books.csv")
        ratings_path = find_file(root, "ratings.csv")
        tags_path = find_file(root, "tags.csv")
        book_tags_path = find_file(root, "book_tags.csv")

        books = pd.read_csv(books_path, encoding="utf-8", on_bad_lines="skip")
        ratings = pd.read_csv(ratings_path, on_bad_lines="skip")
        tags = pd.read_csv(tags_path)
        book_tags = pd.read_csv(book_tags_path)

        return {
            "books": books,
            "ratings": ratings,
            "tags": tags,
            "book_tags": book_tags,
            "source": "goodbooks",
        }

    def _load_amazon(self):
        root = os.path.join(self.raw_dir, "amazon")
        books_path = os.path.join(root, "books_data.csv")
        ratings_path = os.path.join(root, "Books_rating.csv")

        rating_columns = [
            "Title",
            "User_id",
            "review/score",
            "review/text",
            "review/summary",
            "review/time",
        ]

        ratings = pd.read_csv(
            ratings_path,
            usecols=rating_columns,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        books = pd.read_csv(books_path, encoding="utf-8", on_bad_lines="skip")

        return {
            "books": books,
            "ratings": ratings,
            "source": "amazon",
        }
