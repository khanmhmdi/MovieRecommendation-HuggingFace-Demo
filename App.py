import pandas as pd
import streamlit as st
from PIL import Image
import json
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

# with open('./MovieRecommendation/Recommender_System/master_ui/Data/movie_data.json', 'r+', encoding='utf-8') as f:
#     data = json.load(f)
# with open('./MovieRecommendation/Recommender_System/master_ui/Data/movie_titles.json', 'r+', encoding='utf-8') as f:
#     movie_titles = json.load(f)
titles_path = hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems",
                              filename="ContentBase Models/Overview Model/titles.csv")

movie_titles = pd.read_csv(titles_path)
hdr = {'User-Agent': 'Mozilla/5.0'}


def movie_poster_fetcher(imdb_link):
    ## Display Movie Poster
    url_data = requests.get(imdb_link, headers=hdr).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image1")
    movie_poster_link = imdb_dp.attrs['content']
    u = urlopen(movie_poster_link)
    raw_data = u.read()
    image = PIL.Image.open(io.BytesIO(raw_data))
    image = image.resize((158, 301), )
    st.image(image, use_column_width=False)


def get_movie_info(imdb_link):
    url_data = requests.get(imdb_link, headers=hdr).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_content = s_data.find("meta", property="og:description")
    movie_descr = imdb_content.attrs['content']
    movie_descr = str(movie_descr).split('.')
    movie_director = movie_descr[0]
    movie_cast = str(movie_descr[1]).replace('With', 'Cast: ').strip()
    movie_story = 'Story: ' + str(movie_descr[2]).strip() + '.'
    rating = s_data.find("span", class_="sc-bde20123-1 iZlgcd").text
    movie_rating = 'Total Rating count: ' + str(rating)
    return movie_director, movie_cast, movie_story, movie_rating


# def KNN_Movie_Recommender(test_point, k):
#     # Create dummy target variable for the KNN Classifier
#     target = [0 for item in movie_titles]
#     # Instantiate object for the Classifier
#     model = KNearestNeighbours(data, target, test_point, k=k)
#     # Run the algorithm
#     model.fit()
#     # Print list of 10 recommendations < Change value of k for a different number >
#     table = []
#     for i in model.indices:
#         # Returns back movie title and imdb link
#         table.append([movie_titles[i][0], movie_titles[i][2], data[i][-1]])
#     print(table)
#     return table


st.set_page_config(
    page_title="Movie Recommender System",
)


def run():
    img1 = Image.open(hf_hub_download(repo_id="Khanmhmdi/Collaborative-movie-recommendation-systems",
                              filename="logo.jpg"))
    img1 = img1.resize((250, 250), )
    st.image(img1, use_column_width=False)
    st.title("Movie Recommender System")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "Movie Dataset"</h4>''',
                unsafe_allow_html=True)
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    movies = [title[1] for title in movie_titles.values]
    category = ['--Select--', 'Movie based', 'Genre based']
    cat_op = st.selectbox('Select Recommendation Type', category)
    if cat_op == category[0]:
        st.warning('Please select Recommendation Type!!')
    elif cat_op == category[1]:
        select_movie1 = st.selectbox('Select first movie: (Recommendation will be based on these selections)',
                                    ['--Select--'] + movies)
        select_movie2 = st.selectbox('Select second movie: (Recommendation will be based on these selections)',
                                    ['--Select--'] + movies)
        select_movie3 = st.selectbox('Select third movie: (Recommendation will be based on these selections)',
                                    ['--Select--'] + movies)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Fetching a Movie Posters will take a time."</h4>''',
            unsafe_allow_html=True)


        if dec == 'No':
            if select_movie1 == '--Select--' or select_movie2 == '--Select--' or select_movie3 == '--Select--':
                st.warning('Please select three movies!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                # genres1 = data[movies.index(select_movie1)]
                # genres2 = data[movies.index(select_movie2)]
                # genres3 = data[movies.index(select_movie3)]
                # test_points = genres1 + genres2 + genres3
                print("----------------", select_movie1)
                print("----------------", select_movie2)
                print("----------------", select_movie3)
                # print("-----------------",test_points)
                #-----------------------------------------------------
                import RecommendationHandler
                hybrid_Recommendation = RecommendationHandler([select_movie1,select_movie2,select_movie3])
                table = hybrid_Recommendation.hybridRecommendationSystem()
                #-----------------------------------------------------
                # table = KNN_Movie_Recommender(test_points, no_of_reco + 1)
                table.pop(0)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                # for movie, link, ratings in table:
                #     c += 1
                #     director, cast, story, total_rat = get_movie_info(link)
                #     st.markdown(f"({c})[ {movie}]({link})")
                #     st.markdown(director)
                #     st.markdown(cast)
                #     st.markdown(story)
                #     st.markdown(total_rat)
                #     st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
                for i in table:
                    st.markdown(i)
                    c += 1


        else:
            if select_movie1 == '--Select--' or select_movie2 == '--Select--' or select_movie3 == '--Select--':
                st.warning('Please select three movies!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                #-----------------------------------------------------
                from RecommendationHandler import RecommendationHandler
                hybrid_Recommendation = RecommendationHandler([select_movie1,select_movie2,select_movie3])
                table = hybrid_Recommendation.hybridRecommendationSystem()
                #-----------------------------------------------------

                # table = KNN_Movie_Recommender(test_points, no_of_reco + 1)
                table.pop(0)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                # for movie, link, ratings in table:
                #     c += 1
                #     st.markdown(f"({c})[ {movie}]({link})")
                #     movie_poster_fetcher(link)
                #     director, cast, story, total_rat = get_movie_info(link)
                #     st.markdown(director)
                #     st.markdown(cast)
                #     st.markdown(story)
                #     st.markdown(total_rat)
                #     st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
                for i in table:
                    if c > no_of_reco:
                        break
                    st.markdown(f"({c})[ {i}])")
                    c += 1
    elif cat_op == category[2]:
        sel_gen = st.multiselect('Select Genres:', genres)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Fetching a Movie Posters will take a time."</h4>''',
            unsafe_allow_html=True)
        if dec == 'No':
            if sel_gen:
                imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
                no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
                test_point = [1 if genre in sel_gen else 0 for genre in genres]
                test_point.append(imdb_score)
                #-----------------------------------------------------
                import RecommendationHandler
                # hybrid_Recommendation = RecommendationHandler([select_movie1,select_movie2,select_movie3])
                # table = hybrid_Recommendation.hybridRecommendationSystem()
                #-----------------------------------------------------
                # table = KNN_Movie_Recommender(test_point, no_of_reco)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                # for movie, link, ratings in table:
                #     c += 1
                #     st.markdown(f"({c})[ {movie}]({link})")
                #     director, cast, story, total_rat = get_movie_info(link)
                #     st.markdown(director)
                #     st.markdown(cast)
                #     st.markdown(story)
                #     st.markdown(total_rat)
                #     st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
        else:
            if sel_gen:
                imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
                no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
                test_point = [1 if genre in sel_gen else 0 for genre in genres]
                test_point.append(imdb_score)
                #-----------------------------------------------------
                import RecommendationHandler
                # hybrid_Recommendation = RecommendationHandler([select_movie1,select_movie2,select_movie3])
                # table = hybrid_Recommendation.hybridRecommendationSystem()
                # -----------------------------------------------------
                # table = KNN_Movie_Recommender(test_point, no_of_reco)
                # c = 0
                # st.success('Some of the movies from our Recommendation, have a look below')
                # for movie, link, ratings in table:
                #     c += 1
                #     st.markdown(f"({c})[ {movie}]({link})")
                #     movie_poster_fetcher(link)
                #     director, cast, story, total_rat = get_movie_info(link)
                #     st.markdown(director)
                #     st.markdown(cast)
                #     st.markdown(story)
                #     st.markdown(total_rat)
                #     st.markdown('IMDB Rating: ' + str(ratings) + '⭐')


run()
