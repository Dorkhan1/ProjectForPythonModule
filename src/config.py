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

# preparation time cleaning and bins (in minutes)
MAX_MINUTES = 365 * 24 * 60  # 1 year
TIME_BINS = [0, 15, 30, 60, 240, 1440, float("inf")]
TIME_LABELS = ["≤15 min", "15-30 min", "30-60 min", "1-4 h", "4-24 h",
               "> 1 day"]

# number of ingredients bins
INGREDIENT_BINS = [0, 5, 9, 14, float("inf")]
INGREDIENT_LABELS = ["1-5", "6-9", "10-14", "15+"]

# tag groups for category analysis
TAG_GROUPS = {
    "Course": [
        "main-dish", "desserts", "side-dishes", "appetizers", "salads",
        "breads", "breakfast", "soups-stews", "beverages", "snacks",
        "sauces",
    ],
    "Cuisine": [
        "american", "italian", "southern-united-states", "mexican",
        "african", "indian", "greek", "french", "middle-eastern", "chinese",
    ],
    "Main ingredient": [
        "vegetables", "fruit", "chicken", "beef", "cheese", "seafood",
        "pasta", "pork", "potatoes", "beans", "chocolate", "eggs",
    ],
    "Dietary": [
        "low-carb", "healthy", "low-calorie", "vegetarian", "low-fat",
        "vegan", "high-protein", "diabetic", "gluten-free",
    ],
}

# rating scale
NO_RATING = 0  # review without stars
STAR_VALUES = [1, 2, 3, 4, 5]
STAR_COLUMNS = [f"stars_{star}" for star in STAR_VALUES]

# bayesian average: default prior weight = this quantile of n_ratings
PRIOR_WEIGHT_QUANTILE = 0.9

# lower bound of the confidence interval
DEFAULT_CONFIDENCE = 0.95
