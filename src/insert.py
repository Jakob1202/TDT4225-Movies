"""Clean The Movies Dataset and insert it into the MongoDB movies and ratings collections."""

from pathlib import Path

import ast
import argparse

import numpy as np
import pandas as pd

from pymongo import ASCENDING, TEXT

from DbConnector import DbConnector

DATA_DIR = Path(__file__).resolve().parent / "data"
MOVIES_METADATA_FILE = "movies_metadata.csv"
CREDITS_FILE = "credits.csv"
KEYWORDS_FILE = "keywords.csv"

MOVIE_BATCH_SIZE = 5_000
RATINGS_CHUNK_SIZE = 1_000_000          # Ratings read from the CSV and inserted per chunk

SNAPSHOT_DATE = pd.Timestamp("2017-08-01")  # The dataset was collected in July 2017
PLACEHOLDER_MONEY_USD = 1_000           # Budgets/revenues below this are placeholders (e.g. "5" for 5 million)
PLACEHOLDER_OVERVIEWS = {"no overview found.", "no overview", "no movie overview available.", ""}

RATINGS_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}

# Nested columns stored as Python-literal strings, and the type each value should have
NESTED_MOVIE_COLUMNS = {
    "genres": list,
    "production_companies": list,
    "production_countries": list,
    "spoken_languages": list,
    "belongs_to_collection": dict,
}

# Fields kept from every nested object; everything else (poster paths, credit ids, ...) is dropped
CAST_FIELDS = ["id", "name", "character", "gender", "order"]
CREW_FIELDS = ["id", "name", "job", "department", "gender"]


class DataInserter:
    """Load, clean, and insert The Movies Dataset into the movies and ratings collections."""

    def __init__(self, data_dir=DATA_DIR, small=False):
        """Connect to the database and store the insertion settings.

        Args:
            data_dir (Path): Folder containing the CSV files of the dataset.
            small (bool): Use ratings_small.csv and links_small.csv instead of the full files,
                for faster development runs.
        """
        self.data_dir = data_dir
        self.ratings_file = "ratings_small.csv" if small else "ratings.csv"
        self.links_file = "links_small.csv" if small else "links.csv"
        self.connection = DbConnector()
        self.db = self.connection.db

    def run(self) -> None:
        """Run every step needed to clean and insert the data, in order.

        Returns:
            None
        """
        movies = self.load_movies()
        movies = self.remove_malformed_rows(movies)
        movies = self.convert_types(movies)
        movies = self.parse_nested_fields(movies)
        movies = self.remove_duplicate_movies(movies)
        movies = self.clean_invalid_values(movies)

        credits = self.load_credits()
        keywords = self.load_keywords()
        documents = self.build_movie_documents(movies, credits, keywords)
        movie_id_by_movielens_id = self.load_links(set(movies["id"]))
        print(f"Movies ready for insertion: {len(documents):,}")

        self.drop_collections()
        self.insert_movies(documents)
        self.insert_ratings(movie_id_by_movielens_id)
        self.create_indexes()

    # ------------------------------------------------------------------ movies

    def load_movies(self) -> pd.DataFrame:
        """Load movies_metadata.csv.

        The file has columns with mixed types caused by a few malformed rows,
        so it is read with low_memory=False.

        Returns:
            pd.DataFrame: The raw movie metadata.
        """
        df = pd.read_csv(self.data_dir / MOVIES_METADATA_FILE, low_memory=False)
        print(f"Loaded {len(df):,} movies from {MOVIES_METADATA_FILE}")
        return df

    @staticmethod
    def remove_malformed_rows(df) -> pd.DataFrame:
        """Remove rows whose columns are shifted, recognised by a non-numeric id.

        Args:
            df (pd.DataFrame): The raw movie metadata.
        Returns:
            pd.DataFrame: The metadata without malformed rows.
        """
        malformed = pd.to_numeric(df["id"], errors="coerce").isna()
        print(f"Cleaning malformed rows (shifted columns): removed {int(malformed.sum()):,} of {len(df):,}")
        return df.loc[~malformed].copy()

    @staticmethod
    def convert_types(df) -> pd.DataFrame:
        """Convert ids, numbers, dates and flags to their proper types.

        Args:
            df (pd.DataFrame): The movie metadata without malformed rows.
        Returns:
            pd.DataFrame: The metadata with typed columns.
        """
        df = df.copy()
        df["id"] = df["id"].astype(int)
        for column in ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")
        df["adult"] = df["adult"].astype(str).eq("True")
        df["video"] = df["video"].astype(str).eq("True")
        return df

    @staticmethod
    def parse_literal(value, expected_type):
        """Parse a Python-literal string such as "[{'id': 16, 'name': 'Animation'}]".

        The nested fields use single quotes, so they are not valid JSON and are
        parsed with ast.literal_eval.

        Args:
            value (str | float): The raw value; NaN for missing values.
            expected_type (type): list or dict, the type the parsed value should have.
        Returns:
            list | dict | None: The parsed value, or an empty list / None if it is missing or malformed.
        """
        default = [] if expected_type is list else None
        if not isinstance(value, str) or not value.strip():
            return default
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError):
            return default
        return parsed if isinstance(parsed, expected_type) else default

    def parse_nested_fields(self, df) -> pd.DataFrame:
        """Parse the genres, companies, countries, languages and collection of every movie.

        Args:
            df (pd.DataFrame): The typed movie metadata.
        Returns:
            pd.DataFrame: The metadata with lists and dictionaries instead of strings.
        """
        df = df.copy()
        for column, expected_type in NESTED_MOVIE_COLUMNS.items():
            df[column] = df[column].map(lambda value: self.parse_literal(value, expected_type))
        return df

    @staticmethod
    def remove_duplicate_movies(df) -> pd.DataFrame:
        """Keep one row per TMDB id, the one with the highest vote_count.

        Repeated ids only differ in popularity and vote_count, so keeping the
        most-voted row keeps the most up-to-date values.

        Args:
            df (pd.DataFrame): The parsed movie metadata.
        Returns:
            pd.DataFrame: The metadata with one row per movie.
        """
        before = len(df)
        df = (
            df.sort_values("vote_count", ascending=False, kind="mergesort")
            .drop_duplicates(subset="id", keep="first")
            .sort_index()
        )
        print(f"Cleaning duplicate movie ids: removed {before - len(df):,} rows")
        return df

    @staticmethod
    def clean_invalid_values(df) -> pd.DataFrame:
        """Turn values that carry no information into missing values, as decided in EDA.ipynb.

        - Budget and revenue of 0 or below PLACEHOLDER_MONEY_USD mean "unknown".
        - A runtime of 0 means "unknown".
        - The vote_average of a movie without votes is 0, which is not a real score.
        - Release dates after the dataset was collected are not real release dates.
        - Placeholder overviews such as "No overview found." are not real overviews.

        Args:
            df (pd.DataFrame): The movie metadata with one row per movie.
        Returns:
            pd.DataFrame: The metadata with invalid values set to NaN / NaT.
        """
        df = df.copy()
        changes = {
            "budget": df["budget"] < PLACEHOLDER_MONEY_USD,
            "revenue": df["revenue"] < PLACEHOLDER_MONEY_USD,
            "runtime": df["runtime"] == 0,
            "vote_average": df["vote_count"] == 0,
            "release_date": df["release_date"] >= SNAPSHOT_DATE,
            "overview": df["overview"].str.strip().str.lower().isin(PLACEHOLDER_OVERVIEWS),
        }
        for column, invalid in changes.items():
            df[column] = df[column].where(~invalid)
            print(f"Cleaning invalid {column} values: set {int(invalid.sum()):,} to missing")
        return df

    def load_credits(self) -> dict:
        """Load credits.csv and return the cast and crew of every movie.

        Only one row is kept per movie id. The cast is sorted by billing order,
        because the order does not always start at 0, and each person keeps only
        the fields needed for analysis. A gender of 0 means "unknown" and becomes None.

        Returns:
            dict: {movie id: (cast list, crew list)}.
        """
        df = pd.read_csv(self.data_dir / CREDITS_FILE)
        before = len(df)
        df = df.drop_duplicates(subset="id", keep="first")
        print(f"Loaded {before:,} credits rows, removed {before - len(df):,} duplicate movie ids")

        credits = {}
        for movie_id, cast, crew in zip(df["id"], df["cast"], df["crew"]):
            cast = sorted(self.parse_literal(cast, list), key=lambda member: member.get("order", np.inf))
            crew = self.parse_literal(crew, list)
            credits[int(movie_id)] = (
                [self.trim_person(member, CAST_FIELDS) for member in cast],
                [self.trim_person(member, CREW_FIELDS) for member in crew],
            )
        return credits

    @staticmethod
    def trim_person(member, fields) -> dict:
        """Keep the given fields of a cast or crew member and store an unknown gender (0) as None.

        Args:
            member (dict): The parsed cast or crew member.
            fields (list[str]): The fields to keep.
        Returns:
            dict: The trimmed member.
        """
        person = {field: member.get(field) for field in fields}
        if person.get("gender") == 0:
            person["gender"] = None
        return person

    def load_keywords(self) -> dict:
        """Load keywords.csv and return the keywords of every movie.

        Returns:
            dict: {movie id: list of {"id", "name"} keywords}.
        """
        df = pd.read_csv(self.data_dir / KEYWORDS_FILE)
        before = len(df)
        df = df.drop_duplicates(subset="id", keep="first")
        print(f"Loaded {before:,} keyword rows, removed {before - len(df):,} duplicate movie ids")
        return {
            int(movie_id): self.parse_literal(keywords, list)
            for movie_id, keywords in zip(df["id"], df["keywords"])
        }

    @staticmethod
    def to_value(value):
        """Convert a pandas/numpy value into a value MongoDB can store (NaN/NaT become None).

        Args:
            value: The value to convert.
        Returns:
            The converted value.
        """
        if value is None or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, (float, np.floating)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    def build_movie_documents(self, movies, credits, keywords) -> list[dict]:
        """Build one document per movie with its genres, companies, countries, languages,
        collection, keywords, cast and crew embedded.

        Args:
            movies (pd.DataFrame): The cleaned movie metadata.
            credits (dict): {movie id: (cast, crew)} from load_credits.
            keywords (dict): {movie id: keywords} from load_keywords.
        Returns:
            list[dict]: The movie documents, with the TMDB id as _id.
        """
        documents = []
        for row in movies.itertuples(index=False):
            cast, crew = credits.get(row.id, ([], []))
            collection = row.belongs_to_collection
            documents.append({
                "_id": int(row.id),
                "imdb_id": self.to_value(row.imdb_id),
                "title": self.to_value(row.title),
                "original_title": self.to_value(row.original_title),
                "original_language": self.to_value(row.original_language),
                "overview": self.to_value(row.overview),
                "tagline": self.to_value(row.tagline),
                "status": self.to_value(row.status),
                "release_date": self.to_value(row.release_date),
                "runtime": self.to_value(row.runtime),
                "budget": self.to_value(row.budget),
                "revenue": self.to_value(row.revenue),
                "popularity": self.to_value(row.popularity),
                "vote_average": self.to_value(row.vote_average),
                "vote_count": None if pd.isna(row.vote_count) else int(row.vote_count),
                "adult": bool(row.adult),
                "video": bool(row.video),
                "genres": [{"id": genre["id"], "name": genre["name"]} for genre in row.genres],
                "production_companies": [
                    {"id": company["id"], "name": company["name"]} for company in row.production_companies
                ],
                "production_countries": row.production_countries,
                "spoken_languages": row.spoken_languages,
                "belongs_to_collection": (
                    {"id": collection["id"], "name": collection["name"]} if collection else None
                ),
                "keywords": keywords.get(row.id, []),
                "cast": cast,
                "crew": crew,
            })
        return documents

    # ------------------------------------------------------------------ ratings

    def load_links(self, movie_ids) -> dict:
        """Map MovieLens movie ids to TMDB movie ids through the links file.

        Links without a tmdbId, or whose tmdbId is not one of the inserted movies,
        are left out, so their ratings get movie_id = None.

        Args:
            movie_ids (set[int]): The TMDB ids of the inserted movies.
        Returns:
            dict: {MovieLens movieId: TMDB id}.
        """
        links = pd.read_csv(self.data_dir / self.links_file).dropna(subset=["tmdbId"])
        links["tmdbId"] = links["tmdbId"].astype(int)
        known = links["tmdbId"].isin(movie_ids)
        print(
            f"Loaded {len(links):,} links from {self.links_file}, "
            f"{int((~known).sum()):,} point to movies that are not in the metadata"
        )
        return dict(zip(links.loc[known, "movieId"], links.loc[known, "tmdbId"]))

    def insert_ratings(self, movie_id_by_movielens_id) -> None:
        """Read the ratings in chunks, link them to movies and insert them.

        Every rating keeps its MovieLens id and gets the TMDB id of the movie
        (or None when it cannot be linked). The Unix timestamp becomes a date.

        Args:
            movie_id_by_movielens_id (dict): {MovieLens movieId: TMDB id} from load_links.
        Returns:
            None
        """
        inserted = 0
        unlinked = 0
        for chunk in pd.read_csv(self.data_dir / self.ratings_file, dtype=RATINGS_DTYPES, chunksize=RATINGS_CHUNK_SIZE):
            movie_ids = chunk["movieId"].map(movie_id_by_movielens_id)
            unlinked += int(movie_ids.isna().sum())
            # Naive UTC datetimes, which MongoDB stores as UTC dates
            timestamps = pd.to_datetime(chunk["timestamp"], unit="s").dt.to_pydatetime()
            documents = [
                {
                    "user_id": int(user_id),
                    "movie_id": None if pd.isna(tmdb_id) else int(tmdb_id),
                    "movielens_id": int(movielens_id),
                    "rating": float(rating),
                    "timestamp": timestamp,
                }
                for user_id, tmdb_id, movielens_id, rating, timestamp in zip(
                    chunk["userId"], movie_ids, chunk["movieId"], chunk["rating"], timestamps
                )
            ]
            self.db["ratings"].insert_many(documents, ordered=False)
            inserted += len(documents)
            print(f"Inserted {inserted:,} ratings", end="\r")

        print(f"\nInserted {inserted:,} ratings from {self.ratings_file} ({unlinked:,} could not be linked to a movie)")

    # ------------------------------------------------------------------ database

    def drop_collections(self) -> None:
        """Drop the movies and ratings collections so every run starts from empty collections.

        Returns:
            None
        """
        for name in ["movies", "ratings"]:
            self.db[name].drop()
        print("Dropped existing movies and ratings collections")

    def insert_movies(self, documents) -> None:
        """Insert the movie documents in batches.

        Args:
            documents (list[dict]): The movie documents from build_movie_documents.
        Returns:
            None
        """
        for start in range(0, len(documents), MOVIE_BATCH_SIZE):
            self.db["movies"].insert_many(documents[start:start + MOVIE_BATCH_SIZE], ordered=False)
            print(f"Inserted {min(start + MOVIE_BATCH_SIZE, len(documents)):,} movies", end="\r")
        print(f"\nInserted {len(documents):,} movies")

    def create_indexes(self) -> None:
        """Create the indexes used by the queries.

        - A text index on overview and tagline for $text searches.
        - Indexes on the user and movie of each rating, for grouping and $lookup.

        Returns:
            None
        """
        self.db["movies"].create_index([("overview", TEXT), ("tagline", TEXT)], name="overview_tagline_text")
        self.db["ratings"].create_index([("user_id", ASCENDING)])
        self.db["ratings"].create_index([("movie_id", ASCENDING)])
        print("Created text index on movies and indexes on ratings.user_id and ratings.movie_id")


def main():
    """Parse the arguments, then clean and insert the data.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Insert ratings_small.csv and links_small.csv instead of the full ratings (for development)"
    )
    args = parser.parse_args()

    inserter = None
    try:
        inserter = DataInserter(data_dir=args.data_dir, small=args.small)
        inserter.run()
    except Exception as e:
        print("ERROR: Failed to insert data:", e)
        raise
    finally:
        if inserter:
            inserter.connection.close_connection()


if __name__ == "__main__":
    main()
