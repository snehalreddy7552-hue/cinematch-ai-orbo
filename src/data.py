from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import io
import pandas as pd

DATA_URL = (
    "https://files.grouplens.org/datasets/"
    "movielens/ml-latest-small.zip"
)

DATA_DIR = Path("data")
EXTRACT_DIR = DATA_DIR / "ml-latest-small"


def ensure_dataset():
    movies_path = EXTRACT_DIR / "movies.csv"
    ratings_path = EXTRACT_DIR / "ratings.csv"

    if movies_path.exists() and ratings_path.exists():
        return movies_path, ratings_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    request = Request(
        DATA_URL,
        headers={"User-Agent": "CineMatch-AI/1.0"}
    )

    with urlopen(request, timeout=60) as response:
        payload = response.read()

    with ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(DATA_DIR)

    return movies_path, ratings_path


def load_data():
    movies_path, ratings_path = ensure_dataset()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna(
        "(no genres listed)"
    )

    return movies, ratings

