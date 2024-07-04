# movie-pairing-recommender-system

This project implements a hybrid movie recommendation system using content-based and collaborative filtering techniques. It combines data from the MovieLens 1M dataset and IMDB to provide personalized movie recommendations.

## Key Features

- Data preprocessing and merging of MovieLens 1M and IMDB datasets
- Feature extraction using TF-IDF for movie genres and actors
- Collaborative filtering using user ratings
- Hybrid recommendation approach combining content-based and collaborative methods
- Customizable weighting between content-based and collaborative components

## Data Processing

- Cleaned and merged MovieLens 1M and IMDB datasets
- Extracted relevant features including genres, actors, and ratings
- Applied TF-IDF vectorization to movie features

To simplify the process you can use the function `preprocess_datasets()` which will make all the modifications to the datasets and return the used variables.

## Recommendation Algorithm

The system uses a two-step approach:

1. Content-based filtering: Calculates similarity between movies based on genres and actors
2. Collaborative filtering: Identifies similar movies based on user ratings

These approaches are then combined with adjustable weights to produce final recommendations.

## Usage

The main function `make_recommendations_for_users(user1_id, user2_id)` takes two user IDs as input and returns a list of recommended movies. The function:

- Retrieves movies rated by both users
- Creates a joint user profile
- Calculates content-based and collaborative scores
- Combines scores to rank potential recommendations
- Excludes movies already seen by either user

## Future Improvements

- Implement more sophisticated algorithms for better accuracy
- Optimize performance for larger datasets
- Add more features for personalization
- Implement a user interface for easier interaction

This project serves as a foundation for a movie recommendation system and can be extended or adapted for various applications in content recommendation.