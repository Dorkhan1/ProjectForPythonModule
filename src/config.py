"""Project constants: file paths, dataset URL and column schema."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RECIPES_PATH = DATA_DIR / "RAW_recipes.csv"
INTERACTIONS_PATH = DATA_DIR / "RAW_interactions.csv"

DATASET_URL = (
    "https://www.kaggle.com/datasets/shuyangli94/"
    "food-com-recipes-and-user-interactions"
)

# columns to load and their dtypes
RECIPE_DTYPES = {
    "id": "int32",
    "name": "string",
    "minutes": "int64",  # has a 2**31 - 1 outlier
    "tags": "string",  # list stored as string
    "n_steps": "int16",
    "n_ingredients": "int16",
}
RECIPE_DATE_COLUMNS = ["submitted"]

INTERACTION_DTYPES = {
    "user_id": "int64",  # max ~2.0e9
    "recipe_id": "int32",
    "rating": "int8",
}
INTERACTION_DATE_COLUMNS = ["date"]
