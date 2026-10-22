"""Clean The Movies Dataset and insert it into the MongoDB movies and ratings collections.

The cleaning handles missing values, duplicate records, inconsistent values and the references
between the files (see section 24 of EDA.ipynb). Outliers, such as placeholder budgets below
1,000 USD or runtimes above 300 minutes, are not handled yet.
"""

from collections import Counter
from pathlib import Path

import ast
import argparse
import re

import numpy as np
import pandas as pd
import pycountry

from pymongo import ASCENDING, TEXT

from DbConnector import DbConnector

DATA_DIR = Path(__file__).resolve().parent / "data"
MOVIES_METADATA_FILE = "movies_metadata.csv"
CREDITS_FILE = "credits.csv"
KEYWORDS_FILE = "keywords.csv"

MOVIE_BATCH_SIZE = 5_000
RATINGS_BATCH_SIZE = 500_000            # Ratings inserted per insert_many call

RATINGS_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
PLACEHOLDER_OVERVIEWS = {"no overview found.", "no overview", "no movie overview available.", ""}
UNKNOWN_GENDER = 0                      # TMDB gender codes: 0 = unknown, 1 = female, 2 = male

# Nested columns stored as Python-literal strings, and the type each value should have
NESTED_MOVIE_COLUMNS = {
    "genres": list,
    "production_companies": list,
    "production_countries": list,
    "spoken_languages": list,
    "belongs_to_collection": dict,
}

# Fields kept from every cast and crew member; everything else (profile paths, credit ids, ...) is dropped
CAST_FIELDS = ["id", "name", "character", "gender", "order"]
CREW_FIELDS = ["id", "name", "job", "department", "gender"]

# TMDB uses the name "Congo" for two different countries, so both get their own name
COUNTRY_NAMES = {"CD": "Democratic Republic of the Congo", "CG": "Republic of the Congo"}
# Language codes used by TMDB that are not in ISO 639-1
LANGUAGE_NAMES = {"cn": "Cantonese", "xx": "No Language"}


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

        The movies are cleaned first, because the credits, keywords, links and ratings
        may only reference movies that are inserted.

        Returns:
            None
        """
        movies = self.load_movies()
        movies = self.repair_split_records(movies)
        movies = self.convert_types(movies)
        movies = self.parse_nested_fields(movies)
        movies = self.remove_duplicate_movies(movies)
        movies = self.handle_missing_movie_values(movies)
        movies = self.fix_inconsistent_movie_values(movies)

        credits = self.load_credits()
        credits = self.remove_duplicate_credits(credits)
        credits = self.parse_credits(credits)
        credits = self.remove_orphans(credits, movies, "credits")
        credits = self.clean_cast_and_crew(credits)

        keywords = self.load_keywords()
        keywords = self.remove_duplicate_keywords(keywords)
        keywords = self.parse_keywords(keywords)
        keywords = self.remove_orphans(keywords, movies, "keywords")

        links = self.load_links()
        links = self.validate_links(links, movies)

        ratings = self.load_ratings()
        ratings = self.link_ratings_to_movies(ratings, links)
        ratings = self.remove_duplicate_ratings(ratings)

        documents = self.build_movie_documents(movies, credits, keywords)
        print(f"Ready for insertion: {len(documents):,} movies and {len(ratings):,} ratings")

        self.drop_collections()
        self.insert_movies(documents)
        self.insert_ratings(ratings)
        self.create_indexes()
        self.check_referential_integrity()

    # ------------------------------------------------------------------ movies

    def load_movies(self) -> pd.DataFrame:
        """Load movies_metadata.csv with every column as text, so the split records can be repaired
        before the types are converted.

        Returns:
            pd.DataFrame: The raw movie metadata.
        """
        df = pd.read_csv(self.data_dir / MOVIES_METADATA_FILE, dtype=str)
        print(f"Loaded {len(df):,} movies from {MOVIES_METADATA_FILE}")
        return df

    @staticmethod
    def repair_split_records(df) -> pd.DataFrame:
        """Merge the records that were split over two lines in the CSV file.

        The second half of a split record is recognised by a non-numeric id. Its first value
        continues the overview of the row above, and the remaining values belong to the columns
        from popularity onwards, shifted by nine positions (see 24.3 in EDA.ipynb). The values are
        moved back into the first half, which is otherwise empty from popularity onwards, and the
        second half is removed. A non-numeric id without such a first half is removed as well.

        Args:
            df (pd.DataFrame): The raw movie metadata.
        Returns:
            pd.DataFrame: The metadata with the split records merged.
        """
        df = df.copy()
        columns = list(df.columns)
        first_shifted = columns.index("popularity")
        shift = first_shifted - columns.index("belongs_to_collection")
        second_halves = pd.to_numeric(df["id"], errors="coerce").isna()

        repaired = 0
        for index in df.index[second_halves]:
            previous = index - 1
            if previous not in df.index or second_halves[previous] or df.loc[previous, columns[first_shifted:]].notna().any():
                continue
            overview_start = df.loc[previous, "overview"]
            df.loc[previous, "overview"] = ("" if pd.isna(overview_start) else overview_start) + str(df.loc[index, "adult"])
            df.loc[previous, columns[first_shifted:]] = df.loc[index, columns[1:len(columns) - shift]].to_numpy()
            repaired += 1

        df = df.loc[~second_halves]
        print(
            f"Repairing records split over two lines: repaired {repaired:,}, "
            f"removed {int(second_halves.sum()):,} rows without a valid id"
        )
        return df

    @staticmethod
    def convert_types(df) -> pd.DataFrame:
        """Convert ids, numbers, dates and flags to their proper types.

        Values that do not fit the type become missing, e.g. an adult flag that is not True/False.

        Args:
            df (pd.DataFrame): The movie metadata with the split records repaired.
        Returns:
            pd.DataFrame: The metadata with typed columns.
        """
        df = df.copy()
        df["id"] = df["id"].astype(int)
        for column in ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")
        invalid_flags = 0
        for column in ["adult", "video"]:
            flags = df[column].map({"True": True, "False": False})
            invalid_flags += int((flags.isna() & df[column].notna()).sum())
            df[column] = flags.astype("boolean")
        print(f"Converting types: {invalid_flags:,} adult/video values that are not True/False set to missing")
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

        Repeated ids are either exact copies or only differ in popularity and vote_count,
        so keeping the most-voted row keeps the most up-to-date values.

        Args:
            df (pd.DataFrame): The parsed movie metadata.
        Returns:
            pd.DataFrame: The metadata with one row per movie.
        """
        before = len(df)
        repeated_ids = int(df["id"].duplicated(keep=False).sum())
        df = (
            df.sort_values("vote_count", ascending=False, kind="mergesort")
            .drop_duplicates(subset="id", keep="first")
            .sort_index()
        )
        print(f"Cleaning duplicate movie ids: removed {before - len(df):,} rows ({repeated_ids:,} rows shared an id)")
        return df

    @staticmethod
    def handle_missing_movie_values(df) -> pd.DataFrame:
        """Store values that encode "unknown" as missing values.

        - A budget, revenue or runtime of 0 means that the value is unknown.
        - The vote_average of a movie without votes is 0, which is not a real score.
        - Placeholder overviews such as "No overview found." are not real overviews.

        Empty lists (no genres, companies, ...) are kept, since they correctly describe
        that no information is available.

        Args:
            df (pd.DataFrame): The metadata with one row per movie.
        Returns:
            pd.DataFrame: The metadata with implicit missing values set to NaN.
        """
        df = df.copy()
        unknown = {
            "budget": df["budget"].eq(0),
            "revenue": df["revenue"].eq(0),
            "runtime": df["runtime"].eq(0),
            "vote_average": df["vote_count"].eq(0),
            "overview": df["overview"].str.strip().str.lower().isin(PLACEHOLDER_OVERVIEWS),
        }
        for column, mask in unknown.items():
            df[column] = df[column].where(~mask)
            print(f"Handling missing {column} values: set {int(mask.sum()):,} to missing")
        return df

    @staticmethod
    def language_name(code) -> str | None:
        """Return the English name of an ISO 639-1 language code, or None if it is unknown."""
        if code in LANGUAGE_NAMES:
            return LANGUAGE_NAMES[code]
        try:
            language = pycountry.languages.get(alpha_2=code)
        except (KeyError, LookupError):
            language = None
        # Remove qualifiers such as "Modern Greek (1453-)" -> "Modern Greek"
        return re.sub(r"\s*\(.*\)$", "", language.name) if language else None

    def fix_inconsistent_movie_values(self, df) -> pd.DataFrame:
        """Fix values that contradict other values or the meaning of the field.

        - A vote_average of 0 for a movie with votes is impossible (a vote is at least 0.5),
          so it is stored as missing.
        - Spoken languages without a name get the name of their ISO code.
        - "Congo" is the name of two country codes, so both codes get their own name.

        Args:
            df (pd.DataFrame): The metadata with implicit missing values handled.
        Returns:
            pd.DataFrame: The metadata with the inconsistencies fixed.
        """
        df = df.copy()
        impossible_average = df["vote_count"].gt(0) & df["vote_average"].eq(0)
        df["vote_average"] = df["vote_average"].where(~impossible_average)

        named_languages = 0
        def name_languages(languages):
            nonlocal named_languages
            fixed = []
            for language in languages:
                if not str(language.get("name") or "").strip():
                    language = {**language, "name": self.language_name(language.get("iso_639_1"))}
                    named_languages += 1
                fixed.append(language)
            return fixed
        df["spoken_languages"] = df["spoken_languages"].map(name_languages)

        renamed_countries = 0
        def name_countries(countries):
            nonlocal renamed_countries
            fixed = []
            for country in countries:
                if country.get("iso_3166_1") in COUNTRY_NAMES:
                    country = {**country, "name": COUNTRY_NAMES[country["iso_3166_1"]]}
                    renamed_countries += 1
                fixed.append(country)
            return fixed
        df["production_countries"] = df["production_countries"].map(name_countries)

        print(
            f"Fixing inconsistent values: {int(impossible_average.sum()):,} impossible vote averages set to missing, "
            f"{named_languages:,} spoken languages named, {renamed_countries:,} country names made unique"
        )
        return df

    # ------------------------------------------------------------------ credits and keywords

    def load_credits(self) -> pd.DataFrame:
        """Load credits.csv.

        Returns:
            pd.DataFrame: The raw credits with cast and crew as Python-literal strings.
        """
        df = pd.read_csv(self.data_dir / CREDITS_FILE)
        print(f"Loaded {len(df):,} credits rows from {CREDITS_FILE}")
        return df

    @staticmethod
    def remove_duplicate_credits(df) -> pd.DataFrame:
        """Keep one row per movie id.

        Repeated ids are exact copies or contain the same credits in another order,
        so the first row is kept.

        Args:
            df (pd.DataFrame): The raw credits.
        Returns:
            pd.DataFrame: The credits with one row per movie.
        """
        before = len(df)
        exact_copies = int(df.duplicated().sum())
        df = df.drop_duplicates(subset="id", keep="first")
        print(f"Cleaning duplicate credits: removed {before - len(df):,} rows ({exact_copies:,} exact copies)")
        return df

    def parse_credits(self, df) -> pd.DataFrame:
        """Parse the cast and crew lists of every movie.

        Args:
            df (pd.DataFrame): The credits with one row per movie.
        Returns:
            pd.DataFrame: The credits with lists of dictionaries.
        """
        df = df.copy()
        df["id"] = df["id"].astype(int)
        df["cast"] = df["cast"].map(lambda value: self.parse_literal(value, list))
        df["crew"] = df["crew"].map(lambda value: self.parse_literal(value, list))
        return df

    @staticmethod
    def remove_orphans(df, movies, name) -> pd.DataFrame:
        """Remove rows whose movie id does not exist among the cleaned movies.

        Credits and keywords are embedded in the movie documents, so a row without a movie
        would have nothing to belong to.

        Args:
            df (pd.DataFrame): Rows with a movie id in the column "id".
            movies (pd.DataFrame): The cleaned movies.
            name (str): The name of the data, used in the message.
        Returns:
            pd.DataFrame: The rows that reference an existing movie.
        """
        known = df["id"].isin(set(movies["id"]))
        print(f"Checking references from {name} to movies: removed {int((~known).sum()):,} rows of unknown movies")
        return df.loc[known]

    @staticmethod
    def clean_cast_and_crew(df) -> pd.DataFrame:
        """Clean the cast and crew of every movie and keep only the fields needed for analysis.

        - A person has one id in all credits, so an unknown gender (0) is filled with the
          person's known gender from another credit (known genders never conflict), and is
          otherwise stored as None.
        - Every person gets the same name in all credits: the most common spelling.
        - An empty character name is stored as None.
        - A repeated credit of the same actor and character in a movie is removed.
        - The cast is sorted by billing order and re-numbered 0, 1, 2, ..., because the
          original order has gaps and repeated values.

        Args:
            df (pd.DataFrame): The parsed credits.
        Returns:
            pd.DataFrame: The credits with cleaned and trimmed cast and crew lists.
        """
        people = [person for column in ["cast", "crew"] for credits in df[column] for person in credits]
        known_gender = {}
        name_counts = Counter()
        for person in people:
            if person.get("gender", UNKNOWN_GENDER) != UNKNOWN_GENDER:
                known_gender.setdefault(person["id"], person["gender"])
            name_counts[(person["id"], person.get("name"))] += 1
        canonical_name = {}
        for (person_id, name), count in sorted(name_counts.items(), key=lambda item: -item[1]):
            canonical_name.setdefault(person_id, name)

        stats = Counter()

        def clean_person(person, fields):
            cleaned = {field: person.get(field) for field in fields}
            if cleaned["name"] != canonical_name[cleaned["id"]]:
                cleaned["name"] = canonical_name[cleaned["id"]]
                stats["names unified"] += 1
            if cleaned.get("gender", UNKNOWN_GENDER) == UNKNOWN_GENDER:
                cleaned["gender"] = known_gender.get(cleaned["id"])
                stats["genders filled" if cleaned["gender"] is not None else "genders unknown"] += 1
            if "character" in cleaned and not str(cleaned["character"] or "").strip():
                cleaned["character"] = None
                stats["empty characters"] += 1
            return cleaned

        def clean_cast(cast):
            seen = set()
            cleaned = []
            for person in sorted(cast, key=lambda member: member.get("order", np.inf)):
                key = (person.get("id"), person.get("character"))
                if key in seen:
                    stats["repeated cast credits"] += 1
                    continue
                seen.add(key)
                cleaned.append(clean_person(person, CAST_FIELDS))
            if [person["order"] for person in cleaned] != list(range(len(cleaned))):
                stats["casts re-numbered"] += 1
            for order, person in enumerate(cleaned):
                person["order"] = order
            return cleaned

        df = df.copy()
        df["cast"] = df["cast"].map(clean_cast)
        df["crew"] = df["crew"].map(lambda crew: [clean_person(person, CREW_FIELDS) for person in crew])
        print(
            "Cleaning cast and crew: "
            f"{stats['genders filled']:,} unknown genders filled from other credits "
            f"({stats['genders unknown']:,} remain unknown), "
            f"{stats['names unified']:,} name spellings unified, {stats['empty characters']:,} empty characters, "
            f"{stats['repeated cast credits']:,} repeated cast credits removed, {stats['casts re-numbered']:,} casts re-numbered"
        )
        return df

    def load_keywords(self) -> pd.DataFrame:
        """Load keywords.csv.

        Returns:
            pd.DataFrame: The raw keywords as Python-literal strings.
        """
        df = pd.read_csv(self.data_dir / KEYWORDS_FILE)
        print(f"Loaded {len(df):,} keyword rows from {KEYWORDS_FILE}")
        return df

    @staticmethod
    def remove_duplicate_keywords(df) -> pd.DataFrame:
        """Keep one row per movie id; every repeated row is an exact copy.

        Args:
            df (pd.DataFrame): The raw keywords.
        Returns:
            pd.DataFrame: The keywords with one row per movie.
        """
        before = len(df)
        exact_copies = int(df.duplicated().sum())
        df = df.drop_duplicates(subset="id", keep="first")
        print(f"Cleaning duplicate keywords: removed {before - len(df):,} rows ({exact_copies:,} exact copies)")
        return df

    def parse_keywords(self, df) -> pd.DataFrame:
        """Parse the keyword list of every movie and remove keywords without a name.

        Args:
            df (pd.DataFrame): The keywords with one row per movie.
        Returns:
            pd.DataFrame: The keywords as lists of {"id", "name"} dictionaries.
        """
        df = df.copy()
        df["id"] = df["id"].astype(int)
        parsed = df["keywords"].map(lambda value: self.parse_literal(value, list))
        df["keywords"] = parsed.map(
            lambda items: [{"id": item["id"], "name": item["name"]} for item in items if str(item.get("name") or "").strip()]
        )
        removed = int(parsed.map(len).sum() - df["keywords"].map(len).sum())
        print(f"Parsing keywords: removed {removed:,} keywords without a name")
        return df

    # ------------------------------------------------------------------ links and ratings

    def load_links(self) -> pd.DataFrame:
        """Load the links between MovieLens, IMDb and TMDB ids.

        Returns:
            pd.DataFrame: The raw links.
        """
        df = pd.read_csv(self.data_dir / self.links_file)
        print(f"Loaded {len(df):,} links from {self.links_file}")
        return df

    @staticmethod
    def validate_links(df, movies) -> pd.DataFrame:
        """Keep the links that reference an existing movie, so that every rating can reference one.

        A link is valid when its tmdbId belongs to a cleaned movie and its imdbId agrees with
        the imdb_id of that movie (when the movie has one). A link that fails is re-mapped to
        the movie with the same IMDb id if there is one, and is dropped otherwise. This also
        removes the extra MovieLens ids of the TMDB ids that are linked more than once, since
        all but one of them disagree with the movie's IMDb id.

        Args:
            df (pd.DataFrame): The raw links.
            movies (pd.DataFrame): The cleaned movies.
        Returns:
            pd.DataFrame: The valid links with movieId and the TMDB movie_id.
        """
        df = df.copy()
        movie_imdb = pd.to_numeric(movies.set_index("id")["imdb_id"].str.removeprefix("tt"), errors="coerce").dropna()
        movie_by_imdb = pd.Series(movie_imdb.index, index=movie_imdb.to_numpy().astype(int))

        known_movie = df["tmdbId"].isin(set(movies["id"]))
        imdb_of_movie = df["tmdbId"].map(movie_imdb)
        consistent = known_movie & (imdb_of_movie.isna() | imdb_of_movie.eq(df["imdbId"]))
        remapped = df["imdbId"].map(movie_by_imdb).where(~consistent)
        df["movie_id"] = df["tmdbId"].where(consistent, remapped)

        dropped = df["movie_id"].isna()
        print(
            f"Validating links: {int(consistent.sum()):,} valid, {int(remapped.notna().sum()):,} re-mapped through the IMDb id, "
            f"{int(dropped.sum()):,} dropped "
            f"({int((dropped & df['tmdbId'].isna()).sum()):,} without tmdbId, "
            f"{int((dropped & df['tmdbId'].notna() & ~known_movie).sum()):,} with an unknown tmdbId, "
            f"{int((dropped & known_movie).sum()):,} with a disagreeing IMDb id)"
        )
        df = df.dropna(subset=["movie_id"])
        df["movie_id"] = df["movie_id"].astype(int)
        return df[["movieId", "imdbId", "movie_id"]]

    def load_ratings(self) -> pd.DataFrame:
        """Load the ratings with compact types.

        Returns:
            pd.DataFrame: The raw ratings.
        """
        df = pd.read_csv(self.data_dir / self.ratings_file, dtype=RATINGS_DTYPES)
        print(f"Loaded {len(df):,} ratings from {self.ratings_file}")
        return df

    @staticmethod
    def link_ratings_to_movies(df, links) -> pd.DataFrame:
        """Translate the MovieLens id of every rating to the TMDB id of its movie.

        Ratings without a valid link have no movie to reference and are removed, so every
        inserted rating references an existing movie.

        Args:
            df (pd.DataFrame): The raw ratings.
            links (pd.DataFrame): The valid links from validate_links.
        Returns:
            pd.DataFrame: The ratings with the TMDB movie_id.
        """
        df = df.copy()
        df["movie_id"] = df["movieId"].map(links.set_index("movieId")["movie_id"])
        unlinked = df["movie_id"].isna()
        print(f"Linking ratings to movies: removed {int(unlinked.sum()):,} of {len(df):,} ratings without a valid movie")
        df = df.loc[~unlinked]
        df["movie_id"] = df["movie_id"].astype(np.int64)
        return df

    @staticmethod
    def remove_duplicate_ratings(df) -> pd.DataFrame:
        """Keep one rating per user and movie, the latest one.

        After translating MovieLens ids to TMDB ids, two MovieLens ids of the same movie would
        give a user two ratings of one movie. Link validation already prevents this, so this
        step is a safeguard for the (user, movie) key.

        Args:
            df (pd.DataFrame): The ratings with the TMDB movie_id.
        Returns:
            pd.DataFrame: The ratings with one rating per user and movie.
        """
        before = len(df)
        repeated = df.duplicated(subset=["userId", "movie_id"], keep=False)
        if repeated.any():
            # Only the repeated ratings are sorted, so the latest rating of every pair is kept
            latest = df.loc[repeated].sort_values("timestamp", kind="mergesort").drop_duplicates(["userId", "movie_id"], keep="last")
            df = pd.concat([df.loc[~repeated], latest]).sort_index()
        print(f"Cleaning duplicate ratings: removed {before - len(df):,} repeated (user, movie) ratings")
        return df

    # ------------------------------------------------------------------ documents

    @staticmethod
    def to_value(value):
        """Convert a pandas/numpy value into a value MongoDB can store (NaN, NaT and NA become None).

        Args:
            value: The value to convert.
        Returns:
            The converted value.
        """
        if isinstance(value, (list, dict)):
            return value
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        return value

    def build_movie_documents(self, movies, credits, keywords) -> list[dict]:
        """Build one document per movie with its genres, companies, countries, languages,
        collection, keywords, cast and crew embedded.

        Args:
            movies (pd.DataFrame): The cleaned movie metadata.
            credits (pd.DataFrame): The cleaned credits from clean_cast_and_crew.
            keywords (pd.DataFrame): The cleaned keywords from parse_keywords.
        Returns:
            list[dict]: The movie documents, with the TMDB id as _id.
        """
        credits_by_movie = dict(zip(credits["id"], zip(credits["cast"], credits["crew"])))
        keywords_by_movie = dict(zip(keywords["id"], keywords["keywords"]))

        documents = []
        for row in movies.itertuples(index=False):
            cast, crew = credits_by_movie.get(row.id, ([], []))
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
                "adult": self.to_value(row.adult),
                "video": self.to_value(row.video),
                "genres": [{"id": genre["id"], "name": genre["name"]} for genre in row.genres],
                "production_companies": [
                    {"id": company["id"], "name": company["name"]} for company in row.production_companies
                ],
                "production_countries": row.production_countries,
                "spoken_languages": row.spoken_languages,
                "belongs_to_collection": (
                    {"id": collection["id"], "name": collection["name"]} if collection else None
                ),
                "keywords": keywords_by_movie.get(row.id, []),
                "cast": cast,
                "crew": crew,
            })
        return documents

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

    def insert_ratings(self, df) -> None:
        """Insert the ratings in batches.

        Every rating keeps its MovieLens id and references its movie by the TMDB id.
        The Unix timestamp is stored as a date.

        Args:
            df (pd.DataFrame): The cleaned ratings with the TMDB movie_id.
        Returns:
            None
        """
        for start in range(0, len(df), RATINGS_BATCH_SIZE):
            batch = df.iloc[start:start + RATINGS_BATCH_SIZE]
            # Naive UTC datetimes, which MongoDB stores as UTC dates
            timestamps = pd.to_datetime(batch["timestamp"], unit="s").dt.to_pydatetime()
            documents = [
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "movielens_id": int(movielens_id),
                    "rating": float(rating),
                    "timestamp": timestamp,
                }
                for user_id, movie_id, movielens_id, rating, timestamp in zip(
                    batch["userId"], batch["movie_id"], batch["movieId"], batch["rating"], timestamps
                )
            ]
            self.db["ratings"].insert_many(documents, ordered=False)
            print(f"Inserted {min(start + RATINGS_BATCH_SIZE, len(df)):,} ratings", end="\r")
        print(f"\nInserted {len(df):,} ratings")

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

    def check_referential_integrity(self) -> None:
        """Check in the database that every rating references an existing movie.

        MongoDB does not enforce foreign keys, so the references are verified after insertion.

        Returns:
            None
        Raises:
            ValueError: If a rating references a movie that does not exist.
        """
        movie_ids = set(self.db["movies"].distinct("_id"))
        referenced = set(self.db["ratings"].distinct("movie_id"))
        missing = referenced - movie_ids
        print(
            f"Checking referential integrity: ratings reference {len(referenced):,} movies, "
            f"{len(missing):,} of them do not exist"
        )
        if missing:
            raise ValueError(f"{len(missing):,} movies referenced by ratings do not exist")


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
