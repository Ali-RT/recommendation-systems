
import numpy as np


def recommend_movies(user_id, n_movies_to_recommend, model, movies_df, ratings_df):
    # Get user's previous ratings
    user_ratings = ratings_df[ratings_df["userId"] == user_id]

    # Get all movie ids
    all_movie_ids = ratings_df["movieId"].unique()

    # Get movie ids not rated by user
    unrated_movies = np.setdiff1d(all_movie_ids, user_ratings["movieId"])

    # Use MF model to predict ratings for unrated movies
    predicted_ratings = model.predict(user_id, unrated_movies)

    # Get top n movie ids based on predicted ratings
    top_n_idxs = np.argsort(predicted_ratings)[::-1][:n_movies_to_recommend]
    top_n_movie_ids = unrated_movies[top_n_idxs]

    # Look up movie titles
    recommended_movies = movies_df[movies_df["movieId"].isin(top_n_movie_ids)]

    return recommended_movies[["title"]]
