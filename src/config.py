"""Project-wide constants: file paths, dataset source and expected schema.

Keeping these values in one place means the rest of the code never
hard-codes a path or a column name, so changing the data layout only
requires editing this module.
"""

from pathlib import Path

# Project root = parent of the ``src`` package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RECIPES_PATH = DATA_DIR / "RAW_recipes.csv"
INTERACTIONS_PATH = DATA_DIR / "RAW_interactions.csv"

DATASET_URL = (
    "https://www.kaggle.com/datasets/shuyangli94/"
    "food-com-recipes-and-user-interactions"
)

# Columns we actually need, mapped to memory-efficient dtypes.
# Heavy text columns (steps, description, review text) are deliberately
# left out: they are not used for ranking and would triple memory usage.
RECIPE_DTYPES = {
    "id": "int32",
    "name": "string",
    "minutes": "int64",  # contains a 2**31 - 1 outlier, so keep 64 bits
    "tags": "string",  # stringified Python list, parsed later
    "n_steps": "int16",
    "n_ingredients": "int16",
}
RECIPE_DATE_COLUMNS = ["submitted"]

INTERACTION_DTYPES = {
    "user_id": "int64",  # max id is ~2.0e9, too close to the int32 limit
    "recipe_id": "int32",
    "rating": "int8",
}
INTERACTION_DATE_COLUMNS = ["date"]
