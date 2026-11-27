# Food.com Recipe Ranking

A Streamlit web app that ranks the recipes of
[Food.com](https://www.food.com) by how much users like them.

Users rate recipes from 1 to 5 stars, but most recipes have only one or two
ratings. Sorting by the plain average rating puts thousands of recipes
with a single 5-star rating above well-tested recipes with hundreds of
ratings. This project proposes a ranking that takes into account both the
ratings and **how many of them a recipe has**, explains why it beats the
naive average, and explores how the ranking changes across tags,
preparation time and number of ingredients.

## Data

[Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
(Kaggle). Two files are used:

| File | Content | Rows |
|---|---|---:|
| `RAW_recipes.csv` | One row per recipe: name, preparation time, number of steps and ingredients, tags | 231,637 |
| `RAW_interactions.csv` | One row per review: user, recipe, date, rating (0-5) and text | 1,132,367 |

The files are not in the repository (they are larger than GitHub's 100 MB
limit); see [Setup](#setup) to download them.

Main cleaning steps:

- ratings of **0** are reviews without stars (often questions), so they
  are removed (60,847 reviews);
- recipes with no star rating at all are removed (5,047 recipes), leaving
  **226,590 recipes** with **1,071,520 ratings**;
- preparation times of 0 minutes or more than one year are set to missing;
- tags are parsed from text into lists; time and number of ingredients are
  grouped into intervals.

## Ranking methods

| Method | Idea |
|---|---|
| **Naive average** | Mean rating. Ignores the number of ratings. |
| **Bayesian average** | Adds `C` virtual ratings equal to the site average (4.66). `C` is the 90% quantile of the number of ratings per recipe (9). |
| **Lower bound** (main method) | Adds one virtual rating of each star value, then takes the lower end of the 95% confidence interval of the mean rating. Based on Evan Miller, [Ranking Items With Star Ratings](https://www.evanmiller.org/ranking-items-with-star-ratings.html). |

The lower bound is the main method because it rewards recipes that are
rated **high, often and consistently**. The Bayesian average fixes the top
of the ranking, but it ranks an unknown recipe with a single 5-star rating
above a recipe with 518 ratings and an average equal to the site average.

## Main findings

- 110,048 recipes (49%) have a perfect 5.0 average, so the naive ranking
  is mostly noise: its top 100 shares **no recipe** with the other
  methods' top 100.
- Median number of ratings in the top 100: naive **2**, Bayesian **47**,
  lower bound **208**.
- Example: *best banana bread* (1,401 ratings, average 4.82) is ranked
  #120,040 by the naive average, #5,706 by the Bayesian average and **#44**
  by the lower bound.
- Breads, sauces and Italian recipes are over-represented in the top;
  beverages and Indian recipes are under-represented. Preparation time and
  number of ingredients make little difference.
- Across tags, the lift (share in the top / share in all recipes)
  correlates with the number of ratings per recipe (Spearman 0.88) much
  more than with the mean rating (0.19): differences between categories
  mostly reflect **popularity**, not quality.

## Project structure

```
├── app.py                 # Streamlit entry point and page navigation
├── app_pages/             # Streamlit pages (only presentation, no analysis code)
│   ├── shared.py          # cached data loading shared by the pages
│   ├── home.py            # overview and key numbers
│   ├── data.py            # dataset, cleaning steps, rating plots
│   ├── methods.py         # ranking methods, parameters, comparison
│   ├── top_recipes.py     # interactive ranking with filters and CSV download
│   └── categories.py      # analysis by tags, time and ingredients
├── src/                   # project code used by the app
│   ├── config.py          # paths, schema and parameters
│   ├── data_loader.py     # CSV loading with validation and error handling
│   ├── preprocessing.py   # cleaning, aggregation and final recipe table
│   ├── ranking.py         # scoring methods, ranks and method comparison
│   ├── analysis.py        # category summaries, lift, filters
│   ├── plots.py           # matplotlib / seaborn figures
│   └── export.py          # saving results to CSV
├── data/                  # raw CSV files (not tracked)
├── outputs/               # generated CSV files (not tracked)
└── requirements.txt
```

## Libraries

Python 3.13 with:

- **pandas** - loading, cleaning and reshaping the data
- **numpy** and **scipy** - scores, confidence intervals, Spearman correlation
- **matplotlib** and **seaborn** - plots
- **streamlit** - web app

Exact versions are in [`requirements.txt`](requirements.txt).

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/Dorkhan1/ProjectForPythonModule.git
   cd ProjectForPythonModule
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the dataset from
   [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
   and put these two files into a `data/` folder in the project root:

   ```
   data/RAW_recipes.csv
   data/RAW_interactions.csv
   ```

   The other files of the dataset are not needed.

## Usage

Run the web app (with the virtual environment activated):

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. The first load takes a few seconds
while the data is prepared; after that the results are cached.

| Page | Content |
|---|---|
| Overview | Goal of the project and key numbers |
| Data | Dataset, cleaning steps, distribution of ratings |
| Ranking methods | Formulas, adjustable parameters, comparison of the methods |
| Top recipes | Ranking with filters by tag, time, ingredients and number of ratings; CSV download |
| Categories | Lift of each category and best recipes per category |

To save all results as CSV files in `outputs/` without the app:

```bash
python -m src.export
```

If the data files are missing, both the app and the script show a message
with download instructions instead of crashing.
