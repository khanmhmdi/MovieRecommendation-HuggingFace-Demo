# import pandas as pd
# from datetime import datetime
# from fastai import *
# from fastai.collab import *
# from fastai.learner import load_learner
# from fastai.tabular import *
# from pathlib import Path
# import warnings
# import os
#
# pd.set_option('mode.chained_assignment', 'warn')
# # warnings.filterwarnings('ignore')
# base_path = Path('/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/')
#
#
# # learn = load_learner('/home/mkhanmhmdi/Downloads/f5eeba6f8e34569c7d62621be7f798e2bc1461b0b393f3ef00cc270088a7c6f8', cpu=True)
#
#
# def recommendation_story_base(movies) -> list:
#     '''
#     This function will give us the recommendation movies base on the user movies.
#     The return recommendations movies of this function is based on the story of the movies and we call it
#     story base content recommendation system.
#
#     :param movies:
#     :return:
#     '''
#     import torch
#
#     indices = pd.read_csv(
#         '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/indices.csv')
#     cosine_sim = torch.load(
#         '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/cosine_sim.pkl')
#     cosine_sim = cosine_sim.numpy()
#     titles = pd.read_csv(
#         '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/titles.csv')
#     recommendation_movies = get_recommendations_story_base("Super High Me", indices, cosine_sim, titles)
#     print(recommendation_movies)
#
#
# def get_recommendations_story_base(Movie_name, indices, cosine_sim, titles, num_recommendations=50):
#     idx = indices[indices['title'] == Movie_name].index[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:num_recommendations + 1]
#     movie_indices = [i[0] for i in sim_scores]
#     return titles.iloc[movie_indices]['title'].values
#
#
# if __name__ == "__main__":
#     import pickle
#
#     with open('/home/mkhanmhmdi/Downloads/f5eeba6f8e34569c7d62621be7f798e2bc1461b0b393f3ef00cc270088a7c6f8', 'rb') as f:
#         a = pickle.load(f)
#
#     dls = CollabDataLoaders.from_df(pd.read_csv('./ratings_df.csv'), user_name='userId', item_name='original_title',
#                                     rating_name='rating', bs=40960)
#
#     movie_names = list(pd.read_csv('movies_df.csv').drop_duplicates(subset='movieId', keep='first').original_title)
#     size = len(movie_names)
#
#
#     def predict_top_movies(userId, movie_names, dls, learn, count=50):
#         query = {'userId': [userId] * size, 'original_title': movie_names}
#         query_df = pd.DataFrame(data=query)
#         query_dl = dls.test_dl(query_df)
#         preds, y = learn.get_preds(dl=query_dl)
#         results = sorted(zip(preds, movie_names), reverse=True)[:count]
#         for idx, (score, name) in enumerate(results):
#             print("Score: ", round(float(score), 2), " for movie: ", name)
#
#
#     predict_top_movies(100, movie_names, dls, a)