import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

def preprocess_datasets(ml_users, ml_ratings, ml_movies, imdb_name, imdb_title, imdb_rating):
    # Preprocess MovieLens 1M dataset
    ml_users["user_id"] = ml_users["user_id"].apply(lambda x: f"user_{x}")
    ml_users["age_group"] = ml_users["age_group"].apply(lambda x: f"group_{x}")
    ml_users["occupation"] = ml_users["occupation"].apply(lambda x: f"occupation_{x}")

    ml_movies["movie_id"] = ml_movies["movie_id"].apply(lambda x: f"movie_{x}")
    ml_movies["date"] = ml_movies["title"].apply(lambda x: x[-5:-1])
    ml_movies["title"] = ml_movies["title"].apply(lambda x: x[:-7])
    ml_movies["original_title"] = ml_movies["title"].str.extract(r"\((.*)\)").to_string()
    ml_movies["title"] = ml_movies["title"].str.replace(r"\(.*\)", "", regex=True).str.strip()

    ml_movies["title"] = ml_movies["title"].apply(lambda x: "The " + x[:-5] if x[-5:] == ", The" else x)
    ml_movies["title"] = ml_movies["title"].apply(lambda x: "Les " + x[:-5] if x[-5:] == ", Les" else x)

    ml_movies.rename(columns={"title": "primary_title"}, inplace=True)

    ml_ratings["movie_id"] = ml_ratings["movie_id"].apply(lambda x: f"movie_{x}")
    ml_ratings["user_id"] = ml_ratings["user_id"].apply(lambda x: f"user_{x}")
    ml_ratings["rating"] = ml_ratings["rating"].apply(lambda x: float(x))

    # Process titles
    for df in [ml_movies, imdb_title]:
        for col in ['primary_title', 'original_title']:
            df[f'{col}_processed'] = df[col].apply(lambda x: str(x).replace(" ", "").lower())
            df[f'{col}_processed'] = df[f'{col}_processed'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # Hot encode genres
    for df in [ml_movies, imdb_title]:
        genres = set()
        for genre_list in df["genres"].str.split("|"):
            genres.update(genre_list)
        
        for genre in genres:
            df[genre] = df["genres"].apply(lambda gs: int(genre in gs.split("|")))
        
        df.drop(columns=["genres"], inplace=True)

    # Preprocess IMDB dataset
    imdb_title = imdb_title[imdb_title["titleType"].isin(["movie", "short", "tvSeries"])]
    imdb_title.rename(columns={"primaryTitle": "primary_title", "originalTitle": "original_title", "startYear": "date"}, inplace=True)
    imdb_title.drop(columns=["isAdult", "endYear", "runtimeMinutes", "titleType"], inplace=True)

    # Merge datasets
    merged_movies = pd.merge(ml_movies, imdb_title, on=['primary_title_processed', 'date'], how='left')
    merged_movies = merged_movies[~merged_movies["tconst"].isna()]

    # Combine duplicate columns
    x_cols = [col for col in merged_movies.columns if col.endswith('_x')]
    y_cols = [col for col in merged_movies.columns if col.endswith('_y')]

    for x_col in x_cols:
        base_name = x_col[:-2]
        y_col = base_name + '_y'
        
        if y_col in y_cols:
            merged_movies[base_name] = merged_movies[x_col].combine_first(merged_movies[y_col])
            merged_movies.drop(columns=[x_col, y_col], inplace=True)
        else:
            merged_movies.rename(columns={x_col: base_name}, inplace=True)

    for y_col in y_cols:
        if y_col in merged_movies.columns:
            base_name = y_col[:-2]
            merged_movies.rename(columns={y_col: base_name}, inplace=True)

    merged_movies = merged_movies.loc[:, ~merged_movies.columns.duplicated()]

    # Add IMDB ratings
    merged_movies = pd.merge(merged_movies, imdb_rating[['tconst', 'averageRating']], on='tconst', how='left')

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(merged_movies['features'])

    # Create user-item matrix
    user_item_matrix = ml_ratings.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    csr_ratings = csr_matrix(user_item_matrix.values)

    return merged_movies, tfidf_matrix, user_item_matrix, csr_ratings

# Usage:
# merged_movies, tfidf_matrix, user_item_matrix, csr_ratings = preprocess_datasets(ml_users, ml_ratings, ml_movies, imdb_name, imdb_title, imdb_rating)