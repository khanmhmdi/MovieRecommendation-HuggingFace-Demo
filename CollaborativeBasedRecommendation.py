import pandas as pd
from fastai import *
from fastai.collab import *
from fastai.tabular import *
import warnings
import pickle

pd.set_option('mode.chained_assignment', 'warn')
warnings.filterwarnings('ignore')


class CollaborativeBasedRecommendation:
    """
    Collaborative-Based Recommendation System using FastAI collaborative filtering.

    Args:
        rating_df_path (str): Path to the ratings DataFrame CSV file.
        movies_df_path (str): Path to the movies DataFrame CSV file.
        model_path (str): Path to the trained model file.

    Attributes:
        rating_df_path (str): Path to the ratings DataFrame CSV file.
        movies_df_path (str): Path to the movies DataFrame CSV file.
        model_path (str): Path to the trained model file.
        ratings_df (DataFrame): Ratings DataFrame loaded from rating_df_path.
        movies_df (DataFrame): Movies DataFrame loaded from movies_df_path.
        size (int): Size of the movie names list.

    """
    def __init__(self, rating_df_path, movies_df_path, model_path):
        self.model_path = model_path
        self.rating_df_path = rating_df_path
        self.movies_df_path = movies_df_path
        self.ratings_df = pd.read_csv(self.rating_df_path)
        self.movies_df = pd.read_csv(self.movies_df_path)
        self.size = None

    def load_model(self):
        """
        Load the trained model from the model_path.

        """
        with open(self.model_path, 'rb') as f:
            self.learn = pickle.load(f)

    def get_recommendation(self, userid):
        """
        Get movie recommendations for a given user ID.

        Args:
            userid (int): User ID.

        Returns:
            list: List of recommended movie titles.

        """
        self.load_model()
        dls = CollabDataLoaders.from_df(self.ratings_df, user_name='userId', item_name='original_title',
                                        rating_name='rating', bs=40960)
        movie_names = list(self.movies_df.drop_duplicates(subset='movieId', keep='first').original_title)
        self.size = len(movie_names)
        recom_movies = self.predict_top_movies(userid, movie_names, dls, self.learn)
        return recom_movies

    def predict_top_movies(self, userId, movie_names, dls, learn, count=30):
        """
        Predict the top recommended movies for a given user ID.

        Args:
            userId (int): User ID.
            movie_names (list): List of movie names.
            dls (CollabDataLoaders): Collaborative DataLoaders object.
            learn (Learner): FastAI Learner object.
            count (int, optional): Number of top movies to predict. Default is 30.

        Returns:
            list: List of recommended movie titles.

        """
        query = {'userId': [userId] * self.size, 'original_title': movie_names}
        query_df = pd.DataFrame(data=query)
        query_dl = dls.test_dl(query_df)
        preds, y = learn.get_preds(dl=query_dl)
        results = sorted(zip(preds, movie_names), reverse=True)[:count]
        recom_movies = []
        for idx, (score, name) in enumerate(results):
            print("Score: ", round(float(score), 2), " for movie: ", name)
            recom_movies.append(name)
        return list(dict.fromkeys(recom_movies))

    def find_users_seen_only_specific_movies(self, specific_movies):
        """
        Finds the users who have seen only the specific movies or the user who has seen these movies with the minimum movie count.

        Args:
            specific_movies (list): List of specific movie titles to check.

        Returns:
            list: List of user IDs who have seen only the specific movies or the user with the minimum count.

        """
        filtered_df = self.ratings_df[self.ratings_df['original_title'].str.contains('|'.join(specific_movies))]
        print("filtered_df: " , filtered_df)
        user_movie_counts = filtered_df.groupby('userId')['original_title'].nunique().reset_index(name='movie_count')

        users_seen_only_specific_movies = user_movie_counts[user_movie_counts['movie_count'] == len(specific_movies)]['userId'].tolist()

        # if len(users_seen_only_specific_movies) > 0:
        return users_seen_only_specific_movies
        # else:
        #     users_seen_specific_movies = user_movie_counts['userId'].tolist()
        #
        #     user_min_movie_count = min(users_seen_specific_movies, key=lambda user: user_movie_counts[user]['movie_count'])
        #
        #     return [user_min_movie_count]

    def find_users_seen_partial_specific_movies(self, specific_movies):
        """
        Finds the users who have seen at least two or one of the specific movies.

        Args:
            specific_movies (list): List of specific movie titles to check.

        Returns:
            list: List of user IDs who have seen at least two or one of the specific movies.

        """
        filtered_df = self.ratings_df[self.ratings_df['original_title'].isin(specific_movies)]

        user_movie_counts = filtered_df.groupby('userId')['original_title'].nunique().reset_index(
            name='movie_count')

        users_seen_partial_specific_movies = user_movie_counts[user_movie_counts['movie_count'] > 1][
            'userId'].tolist()

        return users_seen_partial_specific_movies


if __name__ == "__main__":
    rating_df_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/ratings_df.csv'
    movies_df_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/movies_df.csv'
    model_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/colab.pth'
    test = CollaborativeBasedRecommendation(rating_df_path, movies_df_path, model_path)
    a = test.find_users_seen_only_specific_movies(['Trois couleurs', 'Les Quatre Cents Coups', 'Sleepless in Seattle'])
    if len(a)==0:
        a = test.find_users_seen_partial_specific_movies(['Trois couleurs', 'Les Quatre Cents Coups', 'Sleepless in Seattle'])
    recom_movies = test.get_recommendation(a[0])
    # print(a)
    # print(recom_movies)
