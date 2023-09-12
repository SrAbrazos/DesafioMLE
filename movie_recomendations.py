import pandas as pd
import re
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='plantillas')

class MovieRecommender:
    
    def __init__(self, movies_path, interactions_path):
        self.movies_df = pd.read_csv(movies_path)
        self.interactions_df = pd.read_csv(interactions_path)
        
    def run_recomendations(self, user_id=None, weeks=8, top_n=10):
        if user_id == None:
            recomendations = self.new_user_recomendations(weeks, top_n)
            return recomendations
        else:
            self.correct_strings_movies()
            plot_matrix, genres_matrix = self.tfidf_vec()
            combined_matrix = self.combined_tfidf(plot_matrix, genres_matrix)
            
            recomendations = self.content_based_user_recommendations(user_id, combined_matrix, top_n)
            return recomendations
            
    def new_user_recomendations(self, weeks, top_n):
        self.interactions_df['date'] = self.interactions_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        current_date = self.interactions_df['date'].max()
        weeks_ago = current_date - timedelta(weeks)
        
        recent_user_ratings = self.interactions_df[self.interactions_df['date'] >= weeks_ago]
        
        movie_popularity_recent = recent_user_ratings.groupby('movieId').size().reset_index(name='recent_view_count')
        
        popular_recent_movies = movie_popularity_recent.sort_values(by='recent_view_count', ascending=False)
        top_popular_recent_movies = popular_recent_movies.head(top_n)
        recommended_movies = pd.merge(top_popular_recent_movies, self.movies_df, on='movieId')
        recommended_movies = recommended_movies['title'].head(top_n).values.tolist()
        
        return recommended_movies
    
    def correct_strings_movies(self):
        self.movies_df['Plot'] = self.movies_df['Plot'].apply(lambda x: re.sub(r'\r\n', '', x).replace('\'', ''))
        self.movies_df['genres'] = self.movies_df['genres'].apply(lambda x: x.replace("|", " "))
        
    def tfidf_vec(self):
        plot_vectorizer = TfidfVectorizer(stop_words='english')
        plot_tfidf = plot_vectorizer.fit_transform(self.movies_df['Plot'].fillna(''))

        genres_vectorizer = TfidfVectorizer(stop_words='english', vocabulary=plot_vectorizer.vocabulary_)
        genres_tfidf = genres_vectorizer.fit_transform(self.movies_df['genres'].fillna(''))
        
        return plot_tfidf, genres_tfidf
    
    @staticmethod
    def combined_tfidf(plot_tfidf, genres_tfidf):
        combined_tfidf_matrix = plot_tfidf + genres_tfidf
        return combined_tfidf_matrix
        

    def content_based_user_recommendations(self, user_id, tfidf_matrix, top_n):
        user_movies = self.interactions_df[self.interactions_df['userId'] == user_id]['movieId'].values
        idx = self.movies_df.loc[self.movies_df['movieId'].isin(user_movies)].index
        
        user_profile = tfidf_matrix[idx].mean(axis=0)
        cosine_similarities = cosine_similarity(user_profile, tfidf_matrix)

        movie_indices = cosine_similarities.argsort()[0][::-1]
        recommended_movies = [self.movies_df['title'][i] for i in movie_indices if i not in user_movies][:top_n]

        return recommended_movies
    
movie_recommender = MovieRecommender('Datos desafío/movies.csv', 'Datos desafío/interactions.csv')

@app.route('/get_new_user_recommendation', methods=['GET'])
def get_new_user_recommendation():
    user = request.args.get('user_id', default=None, type=int)
    weeks = request.args.get('weeks', default=8, type=int)
    top_n = request.args.get('top_n', default=10, type=int)
    recommendations = movie_recommender.run_recomendations(user, weeks, top_n)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)