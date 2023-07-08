import numpy as np
from operator import itemgetter
from ContentBaseOtherFeatures import ContentBaseOtherFeatures
from  ContentBaseMovieStory import ContentBaseMovieStory
from  CollaborativeBasedRecommendation import CollaborativeBasedRecommendation
from huggingface_hub import hf_hub_download


class RecommendationHandler:
    def __init__(self, user_movies):
        self.user_movies = user_movies

    def get_ContentBaseOtherFeatures(self):
        # model_path = './MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/cosine_sim.pkl'
        # data_path = './MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/data.csv'
        # indices_path = './MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/indices.csv'

        model_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/genres_cast_keywords_crew Model/cosine_sim.pkl")
        data_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/genres_cast_keywords_crew Model/data.csv")
        indices_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/genres_cast_keywords_crew Model/indices.csv")

        movie_ContBF_R = ContentBaseOtherFeatures(model_path, data_path, indices_path)
        ContBF_result = []
        for i in self.user_movies:
            try:
                recommendation = movie_ContBF_R.recommendation_genre_cast_keywords_crew(i)
                ContBF_result.extend(recommendation)
            except:
                continue

        return ContBF_result

    def get_ContentBaseMovieStory(self):
        # model_path = './MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/cosine_sim.pkl'
        # titles_path = './MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/titles.csv'
        # indices_path = './MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/indices.csv'

        indices_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/Overview Model/indices.csv")
        titles_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/Overview Model/titles.csv")
        model_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems", filename="ContentBase Models/Overview Model/cosine_sim.pkl")

        movie_ContBMS_R = ContentBaseMovieStory(model_path, titles_path, indices_path)
        ContBMS_results = []
        for i in self.user_movies:
            try:
                recommendation = movie_ContBMS_R.recommendation_story_base(i)
                ContBMS_results.extend(recommendation)
            except:
                continue
        return ContBMS_results

    def get_CollaborativeBasedRecommendation(self):
        # rating_df_path = './MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/ratings_df.csv'
        # movies_df_path = './MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/movies_df.csv'
        # model_path = './MovieRecommendation/Recommender_System/master_ui/Models/Collaborative Model/colab.pth'
        model_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems",
                            filename="Collaborative Model/learners.pkl")
        movies_df_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems",
                            filename="Collaborative Model/movies_df.csv")
        rating_df_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems",
                            filename="Collaborative Model/ratings_df.csv")
        CollBR_movie = CollaborativeBasedRecommendation(rating_df_path, movies_df_path, model_path)
        similar_users = CollBR_movie.find_users_seen_only_specific_movies(self.user_movies)
        if len(similar_users) == 0:
            similar_users = CollBR_movie.find_users_seen_partial_specific_movies(self.user_movies)
        if len(similar_users)==0:
            return []
        recom_movies = CollBR_movie.get_recommendation(similar_users[0])
        return recom_movies

    def hybridRecommendationSystem(self):
        ContBF_recommendation_Result = self.get_ContentBaseOtherFeatures()
        ContBMS_recommendation_Result = self.get_ContentBaseMovieStory()
        CollBR_recommendation_Result = self.get_CollaborativeBasedRecommendation()
        print("ContBF_recommendation_Result")
        print(ContBF_recommendation_Result)
        print("ContBMS_recommendation_Result")
        print(ContBMS_recommendation_Result)
        print("CollBR_recommendation_Result")
        print(CollBR_recommendation_Result)

        find_recommendation = []
        common_movies = self.find_common_movies([ContBF_recommendation_Result,ContBMS_recommendation_Result,CollBR_recommendation_Result])
        common_movies_content_base = self.find_common_movies([ContBF_recommendation_Result,ContBMS_recommendation_Result])
        find_recommendation.extend(common_movies)
        find_recommendation.extend(common_movies_content_base)
        if len(find_recommendation) < 30 :
            find_recommendation.extend(CollBR_recommendation_Result)
        if len(find_recommendation) < 30 :
            find_recommendation.extend(ContBMS_recommendation_Result)

        return list(set(find_recommendation))

    def find_common_movies(self,movies_recommendation):
        result = set(movies_recommendation[0])
        for s in movies_recommendation[1:]:
            result.intersection_update(s)
        print(result)
        return result


