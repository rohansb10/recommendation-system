from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random


app = Flask(__name__)

# Load and process the dataset
df = pd.read_csv('movies.csv')
df.drop(columns=['budget', 'keywords', 'production_companies', 'production_countries'], inplace=True)
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['homepage'] = df['homepage'].fillna('')
df['genres'] = df['genres'].fillna('')
df['combined_features'] = df['genres']+ ' ' + df['overview']+ ' ' + df['tagline'] 

# Compute the TF-IDF matrix and cosine similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

random_image_urls = [
    "static/img/img_1.webp",
    "static/img/img_2.webp",
    "static/img/img_3.webp",
    "static/img/img_4.webp",
    "static/img/img_5.webp",
    "static/img/img_6.webp",
    "static/img/img_7.webp",
    "static/img/img_8.webp",
    "static/img/img_9.webp",
    "static/img/img_10.webp",
    "static/img/img_11.webp",
    "static/img/img_12.webp",
]

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df.index[df['movie name'] == title].tolist()
    if not idx:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9] 
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df.iloc[movie_indices]
    return recommended_movies[['movie name', 'homepage']]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = get_recommendations(movie_title)
        shuffled_images = random.sample(random_image_urls, len(random_image_urls))
        images = shuffled_images[:len(recommendations)]
        movie_data = zip(recommendations['movie name'], recommendations['homepage'], images)
        return render_template('index.html', movie_data=movie_data, movie_title=movie_title)
    return render_template('index.html', movie_data=None, movie_title=None)


if __name__ == '__main__':
    app.run(debug=True)
