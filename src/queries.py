"""Run the Part 2 queries against the movies database and print the results."""

from collections import Counter
from datetime import datetime
from pathlib import Path

import csv
import re

import pandas as pd
import pycountry

from tabulate import tabulate

from DbConnector import DbConnector

TOP_N = 10
PREVIEW_ROWS = 20
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "queries"

MIN_DIRECTOR_MOVIES = 5                 # Task 1
MIN_SHARED_MOVIES = 3                   # Task 2
MIN_ACTOR_MOVIES = 10                   # Task 3
EXAMPLE_GENRES = 5                      # Task 3
MIN_COLLECTION_MOVIES = 3               # Task 4
TOP_BILLED_CAST = 5                     # Task 6
NOIR_SEARCH = "noir"                    # Task 7: "neo-noir" is tokenized into "neo" and "noir", so it matches too
MIN_NOIR_VOTES = 50                     # Task 7
NOIR_LIMIT = 20                         # Task 7
MIN_PAIR_VOTES = 100                    # Task 8
MIN_COLLABORATIONS = 3                  # Task 8
PAIR_LIMIT = 20                         # Task 8
MIN_USER_RATINGS = 20                   # Task 10
NO_LANGUAGE = "xx"                      # Task 9: TMDB code for movies without spoken language

FEMALE = 1                              # TMDB gender codes: 1 = female, 2 = male, unknown stored as None
LANGUAGE_NAME_OVERRIDES = {"cn": "Cantonese", "xx": "No Language"}


def unique_people(people, fields=("id", "name")) -> dict:
    """Build an expression that keeps the first occurrence of every person id in a list.

    An actor can be credited several times in the same movie (several roles), and
    a crew member can have several jobs, so people are made distinct before counting.
    The order of the list is preserved.

    Args:
        people (str | dict): An expression resolving to a list of cast or crew members.
        fields (tuple[str]): The fields kept for every person.
    Returns:
        dict: The MongoDB expression.
    """
    return {"$reduce": {
        "input": people,
        "initialValue": [],
        "in": {"$cond": [
            {"$in": ["$$this.id", "$$value.id"]},
            "$$value",
            {"$concatArrays": ["$$value", [{field: f"$$this.{field}" for field in fields}]]}
        ]}
    }}


def directors_of(crew="$crew") -> dict:
    """Build an expression for the distinct directors (job = Director) of a movie.

    Args:
        crew (str): An expression resolving to the crew list.
    Returns:
        dict: The MongoDB expression.
    """
    return unique_people({"$filter": {"input": crew, "cond": {"$eq": ["$$this.job", "Director"]}}})


def exact_median(values) -> dict:
    """Build an expression for the exact median of the numbers in an array.

    Non-numeric values (missing values stored as None) are ignored. MongoDB's
    $median accumulator is approximate, so the median is computed from the
    sorted array instead.

    Args:
        values (str | dict): An expression resolving to an array of values.
    Returns:
        dict: The MongoDB expression, which resolves to None for an empty array.
    """
    numbers = {"$filter": {"input": values, "cond": {"$isNumber": "$$this"}}}
    return {"$let": {
        "vars": {"sorted": {"$sortArray": {"input": numbers, "sortBy": 1}}},
        "in": {"$let": {
            "vars": {"n": {"$size": "$$sorted"}},
            "in": {"$cond": [
                {"$eq": ["$$n", 0]},
                None,
                {"$avg": [
                    {"$arrayElemAt": ["$$sorted", {"$toInt": {"$floor": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}}]},
                    {"$arrayElemAt": ["$$sorted", {"$toInt": {"$ceil": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}}]}
                ]}
            ]}
        }}
    }}


def count_numbers(field) -> dict:
    """Build an accumulator counting the documents where a field holds a number (not None)."""
    return {"$sum": {"$cond": [{"$isNumber": field}, 1, 0]}}


DECADE = {"$subtract": [{"$year": "$release_date"}, {"$mod": [{"$year": "$release_date"}, 10]}]}


def language_name(code) -> str:
    """Convert an ISO 639-1 language code into a readable name, e.g. "fr" -> "French"."""
    if code in LANGUAGE_NAME_OVERRIDES:
        return LANGUAGE_NAME_OVERRIDES[code]
    try:
        language = pycountry.languages.get(alpha_2=code)
    except (KeyError, LookupError):
        language = None
    return re.sub(r"\s*\(.*\)$", "", language.name) if language else code


class QueryProgram:
    """Run the Part 2 assignment queries and print each result as a table."""

    def __init__(self):
        """Connect to the database and keep the client and database."""
        self.connection = DbConnector()
        self.client = self.connection.client
        self.db = self.connection.db

    def aggregate(self, pipeline, collection="movies", **options) -> list[dict]:
        """Run an aggregation pipeline and return all result documents.

        Args:
            pipeline (list[dict]): The aggregation stages.
            collection (str): The collection to aggregate.
            **options: Extra aggregate options, e.g. hint.
        Returns:
            list[dict]: The result documents.
        """
        return list(self.db[collection].aggregate(pipeline, allowDiskUse=True, **options))

    @staticmethod
    def to_rows(documents, columns) -> tuple[list[list], list[str]]:
        """Turn result documents into table rows.

        Args:
            documents (list[dict]): The result documents.
            columns (dict): {header: field name or function of a document}.
        Returns:
            tuple: The rows and the headers.
        """
        rows = [
            [column(document) if callable(column) else document.get(column) for column in columns.values()]
            for document in documents
        ]
        return rows, list(columns)

    @staticmethod
    def print_header(title):
        """Print a task title between two separator lines.

        Args:
            title (str): The title to print.
        Returns:
            None
        """
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

    @staticmethod
    def print_table(rows, headers, limit=None):
        """Print rows as a table, optionally showing only the first `limit` rows.

        Args:
            rows (list[list]): The rows to print.
            headers (list[str]): The column names.
            limit (int | None): The maximum number of rows to show, or None for all.
        Returns:
            None
        """
        shown = rows if limit is None else rows[:limit]
        # Thousands separators only for floats (revenue), so ids and years stay unformatted
        print(tabulate(shown, headers=headers, tablefmt="psql", floatfmt=",.2f"))
        if limit is not None and len(rows) > limit:
            print(f"... showing {limit} of {len(rows):,} rows")

    @staticmethod
    def save_csv(name, rows, headers):
        """Save all rows of a query result as a CSV file in the results/queries folder.

        Floats are rounded to 4 decimals, dates are written as YYYY-MM-DD and
        lists are joined with "; ".

        Args:
            name (str): The file name without extension, e.g. "task_3".
            rows (list[list]): The rows to save.
            headers (list[str]): The column names, written as the first row.
        Returns:
            None
        """
        def to_csv_value(value):
            if isinstance(value, float):
                return round(value, 4)
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d")
            if isinstance(value, list):
                return "; ".join(map(str, value))
            return value

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(to_csv_value(value) for value in row)
        print(f"Saved {len(rows):,} rows to {path.relative_to(RESULTS_DIR.parents[1])}")

    def task_1(self):
        """List the 10 directors with at least 5 movies that have the highest median revenue,
        with their movie count and mean vote_average.

        The median only uses movies with a known revenue (unknown revenues are stored
        as None), and directors without any known revenue are left out.

        Returns:
            None
        """
        self.print_header(
            f"1. Top {TOP_N} directors (≥ {MIN_DIRECTOR_MOVIES} movies) by median revenue"
        )
        documents = self.aggregate([
            {"$project": {"revenue": 1, "vote_average": 1, "directors": directors_of()}},
            {"$unwind": "$directors"},
            {"$group": {
                "_id": "$directors.id",
                "director": {"$first": "$directors.name"},
                "movies": {"$sum": 1},
                "movies_with_revenue": count_numbers("$revenue"),
                "revenues": {"$push": "$revenue"},
                "mean_vote_average": {"$avg": "$vote_average"},
            }},
            {"$match": {"movies": {"$gte": MIN_DIRECTOR_MOVIES}, "movies_with_revenue": {"$gt": 0}}},
            {"$set": {"median_revenue": exact_median("$revenues")}},
            {"$sort": {"median_revenue": -1, "movies": -1}},
            {"$limit": TOP_N},
        ])
        rows, headers = self.to_rows(documents, {
            "director": "director",
            "movies": "movies",
            "movies_with_revenue": "movies_with_revenue",
            "median_revenue_usd": "median_revenue",
            "mean_vote_average": "mean_vote_average",
        })
        self.print_table(rows, headers)
        self.save_csv("task_1", rows, headers)

    def task_2(self):
        """List the actor pairs that co-starred in at least 3 movies, with their number of
        co-appearances and the average vote_average of these movies, sorted by co-appearances.

        Every pair is counted once per movie, with the lower person id first.

        Returns:
            None
        """
        self.print_header(f"2. Actor pairs that co-starred in ≥ {MIN_SHARED_MOVIES} movies")
        documents = self.aggregate([
            {"$project": {"vote_average": 1, "actors": unique_people("$cast")}},
            {"$match": {"actors.1": {"$exists": True}}},
            {"$set": {"actor_a": "$actors"}},
            {"$unwind": "$actor_a"},
            {"$project": {
                "vote_average": 1,
                "actor_a": 1,
                "actor_b": {"$filter": {"input": "$actors", "cond": {"$gt": ["$$this.id", "$actor_a.id"]}}},
            }},
            {"$unwind": "$actor_b"},
            {"$group": {
                "_id": {"a": "$actor_a.id", "b": "$actor_b.id"},
                "actor_1": {"$first": "$actor_a.name"},
                "actor_2": {"$first": "$actor_b.name"},
                "co_appearances": {"$sum": 1},
                "avg_vote_average": {"$avg": "$vote_average"},
            }},
            {"$match": {"co_appearances": {"$gte": MIN_SHARED_MOVIES}}},
            {"$sort": {"co_appearances": -1, "avg_vote_average": -1, "actor_1": 1, "actor_2": 1}},
        ])
        rows, headers = self.to_rows(documents, {
            "actor_1": "actor_1",
            "actor_2": "actor_2",
            "co_appearances": "co_appearances",
            "avg_vote_average": "avg_vote_average",
        })
        print(f"{len(rows):,} actor pairs co-starred in at least {MIN_SHARED_MOVIES} movies")
        self.print_table(rows, headers, limit=PREVIEW_ROWS)
        self.save_csv("task_2", rows, headers)

    def task_3(self):
        """List the 10 actors (with at least 10 credited movies) with the widest genre breadth:
        the number of distinct genres and up to 5 example genres.

        Ties in genre breadth are broken by the number of movies. The example genres
        are the actor's most frequent genres, counted in Python from the genre lists.

        Returns:
            None
        """
        self.print_header(
            f"3. Top {TOP_N} actors (≥ {MIN_ACTOR_MOVIES} movies) with the widest genre breadth"
        )
        documents = self.aggregate([
            {"$project": {"genres": "$genres.name", "actors": unique_people("$cast")}},
            {"$unwind": "$actors"},
            {"$group": {
                "_id": "$actors.id",
                "actor": {"$first": "$actors.name"},
                "movies": {"$sum": 1},
                "genre_lists": {"$push": "$genres"},
            }},
            {"$match": {"movies": {"$gte": MIN_ACTOR_MOVIES}}},
            {"$set": {"genres": {"$reduce": {
                "input": "$genre_lists", "initialValue": [], "in": {"$concatArrays": ["$$value", "$$this"]}
            }}}},
            {"$set": {"distinct_genres": {"$size": {"$setUnion": ["$genres", []]}}}},
            {"$sort": {"distinct_genres": -1, "movies": -1, "actor": 1}},
            {"$project": {"genre_lists": 0}},
        ])
        top_breadth = documents[0]["distinct_genres"]
        tied = sum(document["distinct_genres"] == top_breadth for document in documents)
        rows, headers = self.to_rows(documents[:TOP_N], {
            "actor": "actor",
            "movies": "movies",
            "distinct_genres": "distinct_genres",
            "example_genres": lambda document: ", ".join(
                genre for genre, _ in Counter(document["genres"]).most_common(EXAMPLE_GENRES)
            ),
        })
        print(
            f"{len(documents):,} actors have at least {MIN_ACTOR_MOVIES} movies; "
            f"{tied:,} of them share the widest breadth of {top_breadth} genres (ties broken by movie count)"
        )
        self.print_table(rows, headers)
        self.save_csv("task_3", rows, headers)

    def task_4(self):
        """List the 10 collections (with at least 3 movies) with the largest total revenue:
        movie count, total revenue, median vote_average and earliest → latest release date.

        The total revenue only includes movies with a known revenue, so it is a lower bound.

        Returns:
            None
        """
        self.print_header(
            f"4. Top {TOP_N} collections (≥ {MIN_COLLECTION_MOVIES} movies) by total revenue"
        )
        documents = self.aggregate([
            {"$match": {"belongs_to_collection.name": {"$ne": None}}},
            {"$group": {
                "_id": "$belongs_to_collection.name",
                "movies": {"$sum": 1},
                "movies_with_revenue": count_numbers("$revenue"),
                "total_revenue": {"$sum": "$revenue"},
                "vote_averages": {"$push": "$vote_average"},
                "first_release": {"$min": "$release_date"},
                "last_release": {"$max": "$release_date"},
            }},
            {"$match": {"movies": {"$gte": MIN_COLLECTION_MOVIES}}},
            {"$set": {"median_vote_average": exact_median("$vote_averages")}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": TOP_N},
        ])
        rows, headers = self.to_rows(documents, {
            "collection": "_id",
            "movies": "movies",
            "movies_with_revenue": "movies_with_revenue",
            "total_revenue_usd": "total_revenue",
            "median_vote_average": "median_vote_average",
            "release_dates": lambda document: (
                f"{document['first_release']:%Y-%m-%d} → {document['last_release']:%Y-%m-%d}"
            ),
        })
        self.print_table(rows, headers)
        self.save_csv("task_4", rows, headers)

    def task_5(self):
        """Compute the median runtime and movie count per decade and primary genre
        (first element in genres), sorted by decade and then median runtime (desc).

        Movies without a release date or without genres cannot be placed and are left
        out. The median only uses known runtimes (runtime 0 is stored as None).

        Returns:
            None
        """
        self.print_header("5. Median runtime and movie count per decade and primary genre")
        documents = self.aggregate([
            {"$match": {"release_date": {"$ne": None}, "genres.0": {"$exists": True}}},
            {"$group": {
                "_id": {"decade": DECADE, "primary_genre": {"$arrayElemAt": ["$genres.name", 0]}},
                "movies": {"$sum": 1},
                "movies_with_runtime": count_numbers("$runtime"),
                "runtimes": {"$push": "$runtime"},
            }},
            {"$set": {"median_runtime": exact_median("$runtimes")}},
            {"$sort": {"_id.decade": 1, "median_runtime": -1, "_id.primary_genre": 1}},
        ])
        rows, headers = self.to_rows(documents, {
            "decade": lambda document: f"{document['_id']['decade']}s",
            "primary_genre": lambda document: document["_id"]["primary_genre"],
            "movies": "movies",
            "movies_with_runtime": "movies_with_runtime",
            "median_runtime_min": "median_runtime",
        })
        print(f"{len(rows):,} (decade, primary genre) groups")
        self.print_table(rows, headers)
        self.save_csv("task_5", rows, headers)

    def task_6(self):
        """Compute, per decade, the average proportion of women among the top 5 billed
        cast members of each movie, sorted by that proportion (desc).

        The top-billed cast is the first 5 distinct people sorted by billing order.
        Unknown genders are ignored, so the proportion is female / (female + male),
        and movies where no top-billed gender is known are left out.

        Returns:
            None
        """
        self.print_header(f"6. Proportion of women among the top {TOP_BILLED_CAST} billed cast members per decade")
        top_billed = {"$slice": [
            unique_people({"$sortArray": {"input": "$cast", "sortBy": {"order": 1}}}, fields=("id", "gender")),
            TOP_BILLED_CAST,
        ]}
        documents = self.aggregate([
            {"$match": {"release_date": {"$ne": None}}},
            {"$project": {"decade": DECADE, "top_billed": top_billed}},
            {"$set": {
                "known": {"$filter": {"input": "$top_billed", "cond": {"$ne": ["$$this.gender", None]}}},
                "female": {"$filter": {"input": "$top_billed", "cond": {"$eq": ["$$this.gender", FEMALE]}}},
            }},
            {"$match": {"known.0": {"$exists": True}}},
            {"$group": {
                "_id": "$decade",
                "avg_female_proportion": {"$avg": {"$divide": [{"$size": "$female"}, {"$size": "$known"}]}},
                "movies": {"$sum": 1},
                "known_cast_members": {"$sum": {"$size": "$known"}},
            }},
            {"$sort": {"avg_female_proportion": -1}},
        ])
        rows, headers = self.to_rows(documents, {
            "decade": lambda document: f"{document['_id']}s",
            "avg_female_proportion_pct": lambda document: 100 * document["avg_female_proportion"],
            "movies": "movies",
            "known_gender_cast_members": "known_cast_members",
        })
        self.print_table(rows, headers)
        self.save_csv("task_6", rows, headers)

    def task_7(self):
        """List the 20 movies with the highest vote_average (vote_count ≥ 50) whose overview
        or tagline matches "noir" or "neo-noir" in a text search.

        The text index on overview and tagline tokenizes "neo-noir" into "neo" and
        "noir", so searching for "noir" finds both forms.

        Returns:
            None
        """
        self.print_header(
            f'7. Top {NOIR_LIMIT} "noir" / "neo-noir" movies (vote_count ≥ {MIN_NOIR_VOTES}) by vote_average'
        )
        matches = self.db["movies"].count_documents({"$text": {"$search": NOIR_SEARCH}})
        documents = self.aggregate([
            {"$match": {"$text": {"$search": NOIR_SEARCH}, "vote_count": {"$gte": MIN_NOIR_VOTES}}},
            {"$sort": {"vote_average": -1, "vote_count": -1}},
            {"$limit": NOIR_LIMIT},
            {"$project": {"title": 1, "year": {"$year": "$release_date"}, "vote_average": 1, "vote_count": 1}},
        ])
        rows, headers = self.to_rows(documents, {
            "title": "title",
            "year": "year",
            "vote_average": "vote_average",
            "vote_count": "vote_count",
        })
        print(
            f"{matches:,} movies match the text search; {len(rows):,} of them have at least "
            f"{MIN_NOIR_VOTES} votes"
        )
        self.print_table(rows, headers)
        self.save_csv("task_7", rows, headers)

    def task_8(self):
        """List the 20 director–actor pairs with at least 3 collaborations that have the
        highest mean vote_average, using only movies with vote_count ≥ 100, together with
        the number of films and their mean revenue.

        A director acting in their own movie is not counted as a pair, since a pair
        consists of two different people. Each pair is counted once per movie, and the
        mean revenue only uses films with a known revenue.

        Returns:
            None
        """
        self.print_header(
            f"8. Top {PAIR_LIMIT} director–actor pairs (≥ {MIN_COLLABORATIONS} films, "
            f"vote_count ≥ {MIN_PAIR_VOTES}) by mean vote_average"
        )
        documents = self.aggregate([
            {"$match": {"vote_count": {"$gte": MIN_PAIR_VOTES}}},
            {"$project": {
                "vote_average": 1,
                "revenue": 1,
                "directors": directors_of(),
                "actors": unique_people("$cast"),
            }},
            {"$unwind": "$directors"},
            {"$unwind": "$actors"},
            {"$match": {"$expr": {"$ne": ["$directors.id", "$actors.id"]}}},
            {"$group": {
                "_id": {"director": "$directors.id", "actor": "$actors.id"},
                "director": {"$first": "$directors.name"},
                "actor": {"$first": "$actors.name"},
                "films": {"$sum": 1},
                "mean_vote_average": {"$avg": "$vote_average"},
                "films_with_revenue": count_numbers("$revenue"),
                "mean_revenue": {"$avg": "$revenue"},
            }},
            {"$match": {"films": {"$gte": MIN_COLLABORATIONS}}},
            {"$sort": {"mean_vote_average": -1, "films": -1, "director": 1, "actor": 1}},
            {"$limit": PAIR_LIMIT},
        ])
        rows, headers = self.to_rows(documents, {
            "director": "director",
            "actor": "actor",
            "films": "films",
            "mean_vote_average": "mean_vote_average",
            "films_with_revenue": "films_with_revenue",
            "mean_revenue_usd": "mean_revenue",
        })
        self.print_table(rows, headers)
        self.save_csv("task_8", rows, headers)

    def task_9(self):
        """Find the 10 most common original languages among non-English movies with the
        United States as production country, with the count and one example title.

        Production companies only have a name and an id in the dataset, not a country,
        so US involvement is determined from production_countries. The TMDB code "xx"
        (No Language, e.g. silent or wordless movies) is not a language and is left out.
        The example title is the most-voted movie of each language.

        Returns:
            None
        """
        self.print_header(f"9. Top {TOP_N} original languages of non-English movies produced in the United States")
        documents = self.aggregate([
            {"$match": {
                "original_language": {"$nin": [None, "en", NO_LANGUAGE]},
                "production_countries.iso_3166_1": "US",
            }},
            {"$sort": {"vote_count": -1}},
            {"$group": {
                "_id": "$original_language",
                "movies": {"$sum": 1},
                "example_title": {"$first": "$title"},
            }},
            {"$sort": {"movies": -1, "_id": 1}},
            {"$limit": TOP_N},
        ])
        rows, headers = self.to_rows(documents, {
            "language_code": "_id",
            "language": lambda document: language_name(document["_id"]),
            "movies": "movies",
            "example_title": "example_title",
        })
        self.print_table(rows, headers)
        self.save_csv("task_9", rows, headers)

    def task_10(self):
        """Compute the ratings count, population variance of ratings and number of distinct
        genres rated for every user, then list the 10 most genre-diverse users and,
        separately, the 10 users with the highest variance (users with ≥ 20 ratings).

        The counts and variances come from all ratings. The genres are looked up
        through the movie each rating references, so ratings that could not be linked
        to a movie do not add genres. Ties are broken by the number of ratings.

        Returns:
            None
        """
        self.print_header(f"10. Rating statistics per user (users with ≥ {MIN_USER_RATINGS} ratings)")
        statistics = pd.DataFrame(self.aggregate([
            {"$group": {
                "_id": "$user_id",
                "ratings": {"$sum": 1},
                "rating_std": {"$stdDevPop": "$rating"},
            }},
            {"$match": {"ratings": {"$gte": MIN_USER_RATINGS}}},
        ], collection="ratings")).set_index("_id")

        # Group the ratings by movie first, so the genres are looked up once per movie
        # (about 45,000 lookups) instead of once per rating (26 million lookups)
        genres = pd.DataFrame(self.aggregate([
            {"$group": {"_id": "$movie_id", "users": {"$push": "$user_id"}}},
            {"$match": {"_id": {"$ne": None}}},
            {"$lookup": {
                "from": "movies",
                "localField": "_id",
                "foreignField": "_id",
                "pipeline": [{"$project": {"_id": 0, "genres": "$genres.name"}}],
                "as": "movie",
            }},
            {"$unwind": "$movie"},
            {"$unwind": "$movie.genres"},
            {"$unwind": "$users"},
            {"$group": {"_id": {"user": "$users", "genre": "$movie.genres"}}},
            {"$group": {"_id": "$_id.user", "distinct_genres": {"$sum": 1}}},
        # Read the ratings in storage order: grouping through the movie_id index would
        # fetch the 26 million ratings in random order, which is many times slower
        ], collection="ratings", hint={"$natural": 1})).set_index("_id")

        users = statistics.join(genres, how="left").fillna({"distinct_genres": 0})
        users["rating_variance"] = users["rating_std"] ** 2
        users["distinct_genres"] = users["distinct_genres"].astype(int)
        users = users.rename_axis("user_id").reset_index()
        columns = ["user_id", "ratings", "rating_variance", "distinct_genres"]
        print(f"{len(users):,} users have at least {MIN_USER_RATINGS} ratings")

        most_diverse = users.sort_values(
            ["distinct_genres", "ratings", "user_id"], ascending=[False, False, True]
        ).head(TOP_N)
        tied = int((users["distinct_genres"] == users["distinct_genres"].max()).sum())
        print(f"\nTop {TOP_N} most genre-diverse users ({tied:,} users rated all {users['distinct_genres'].max()} genres)")
        rows = most_diverse[columns].astype(object).values.tolist()
        self.print_table(rows, columns)
        self.save_csv("task_10_genre_diversity", rows, columns)

        highest_variance = users.sort_values(
            ["rating_variance", "ratings", "user_id"], ascending=[False, False, True]
        ).head(TOP_N)
        print(f"\nTop {TOP_N} users with the highest rating variance")
        rows = highest_variance[columns].astype(object).values.tolist()
        self.print_table(rows, columns)
        self.save_csv("task_10_variance", rows, columns)


def main():
    """Connect to the database, run every task in order, and close the connection.

    Returns:
        None
    """
    program = None
    try:
        program = QueryProgram()
        program.task_1()
        program.task_2()
        program.task_3()
        program.task_4()
        program.task_5()
        program.task_6()
        program.task_7()
        program.task_8()
        program.task_9()
        program.task_10()
    except Exception as e:
        print("ERROR: Failed to use database:", e)
    finally:
        if program:
            program.connection.close_connection()


if __name__ == '__main__':
    main()
