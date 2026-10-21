# TDT4225 Movies

This project contains the Python and MongoDB work for Assignment 3 in TDT4225 - Very Large, Distributed Data Volumes. It uses the Movies Dataset from Kaggle and a MongoDB 7.0.1 database.

## Project Structure

```text
.
├── compose.yaml                # MongoDB 7.0.1 service definition
├── docker/mongo-init.js        # Creates the readWrite database user on first start
├── pyproject.toml              # Project dependencies and Python version
├── requirements.txt            # Assignment dependencies
├── src/
│   ├── EDA.ipynb               # Exploratory data analysis and data-quality checks
│   ├── insert.py               # Cleans and inserts data into MongoDB
│   ├── queries.py              # Runs the Part 2 assignment queries
│   ├── DbConnector.py          # Connects Python to the MongoDB database
│   ├── data/*.csv              # Movies Dataset
│   ├── results/figures/*.png   # Analysis results
│   └── results/queries/*.csv   # Queries results
└── .env                        # Local database passwords; do not commit secrets
```

## Requirements

- Python 3.13 or newer
- `uv`
- Docker Desktop or another Docker installation

## Setup

Run these commands from the project root.

### 1. Install the Python environment

```bash
uv sync --locked
```

This creates or updates the project virtual environment from `uv.lock` and installs the required dependencies.

### 2. Configure database credentials

Create or update `.env` in the project root:

```dotenv
MONGO_ROOT_PASSWORD=choose_a_root_password
MONGO_PASSWORD=choose_a_database_password
```

The database user created by Docker is `movies`, the database is `movies`, and the database password is read from `MONGO_PASSWORD`. The root user is `root` with the password from `MONGO_ROOT_PASSWORD`.

### 3. Start MongoDB

Start the database with Docker Compose:

```bash
docker compose up -d
```

The MongoDB server is available at `127.0.0.1:27017`. Compose uses the MongoDB `7.0.1` image declared in `compose.yaml`; no separate `docker pull` command is required because Docker pulls the image automatically when needed. On the first start, `docker/mongo-init.js` creates the `movies` user with the `readWrite` role on the `movies` database.

Check that the container is running:

```bash
docker compose ps
```

To stop the database without deleting its data:

```bash
docker compose down
```

### 4. Upload Dataset

Download the Movies Dataset from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and upload the CSV files to the `data` folder:

```text
src/
└── data/
    ├── credits.csv
    ├── keywords.csv
    ├── links.csv
    ├── links_small.csv
    ├── movies_metadata.csv
    ├── ratings.csv
    └── ratings_small.csv
```

Make sure the files keep their original names. `ratings_small.csv` and `links_small.csv` may be used during development, but the final solution must handle the full `ratings.csv` and `links.csv`.

## Assignment Workflow

### Part 1: EDA, cleaning, and insertion

1. Open `src/EDA.ipynb` to inspect the dataset and perform exploratory data analysis. The notebook examines the data distribution, nested JSON fields, missing values, and other data-quality issues. Its figures and results are stored in `src/results/figures`.
2. Run `src/insert.py` to clean the dataset and insert the prepared documents into the MongoDB database.
3. The collection design and cleaning decisions should be documented in the report, including the choice between embedded documents and references.

### Part 2: Database queries

Run `src/queries.py` after the data has been inserted. It handles the assignment queries, including director revenue, actor co-appearances, actor genre breadth, collection revenue, runtime by decade and genre, female cast proportion, noir text search, director–actor pairs, non-English US productions, and user rating diversity. The results are stored in `src/results/queries`.

## Running Python Files

Because the project uses `uv`, run Python commands from the project root with:

```bash
uv run python src/insert.py
uv run python src/queries.py
```

`DbConnector.py` loads `MONGO_PASSWORD` from `.env` and connects to the MongoDB database running in Docker. Start the database before running scripts that use the connector.

## Troubleshooting

- If the connection fails, verify that Docker is running and that `docker compose ps` shows the MongoDB service as running.
- If credentials were changed after the database was first started, remove the existing Docker volume and start again. This deletes the local database data:

  ```bash
  docker compose down -v
  docker compose up -d
  ```

- Do not commit `.env` or real database passwords to the repository.
