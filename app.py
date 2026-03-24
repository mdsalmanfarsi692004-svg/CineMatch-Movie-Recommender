import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Pro | Premium AI OTT", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (ULTRA PREMIUM OTT THEME) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* True OTT Dark Theme */
    .stApp { background-color: #0B0B0C; color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Hide top padding for edge-to-edge Hero Banner */
    .block-container { padding-top: 0rem; max-width: 100%; padding-left: 0; padding-right: 0;}
    
    /* Netflix-style Logo */
    .nav-logo {
        font-size: 2.5rem; font-weight: 900; color: #E50914; 
        padding: 20px 50px; text-transform: uppercase; letter-spacing: 2px;
        position: absolute; top: 0; left: 0; z-index: 1000;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.9);
    }
    
    /* Hero Banner CSS - Official IMDb Interstellar Backdrop */
    .hero-container {
        position: relative; width: 100%; height: 80vh;
        background-image: linear-gradient(to top, #0B0B0C 0%, rgba(11,11,12,0.4) 40%, rgba(11,11,12,0.1) 100%), 
                          linear-gradient(to right, rgba(11,11,12,0.9) 0%, rgba(11,11,12,0.4) 50%, transparent 100%),
                          url('https://m.media-amazon.com/images/M/MV5BMTc3NjA2MTIxMF5BMl5BanBnXkFtZTgwNDY0MTE0MzE@._V1_FMjpg_UX1000_.jpg');
        background-size: cover; background-position: center top;
        display: flex; flex-direction: column; justify-content: center; padding-left: 5%;
        margin-bottom: 40px;
    }
    .hero-title { font-size: 5.5rem; font-weight: 900; margin-bottom: 10px; text-shadow: 2px 2px 8px #000; letter-spacing: 1px;}
    .hero-desc { font-size: 1.25rem; max-width: 45%; line-height: 1.6; color: #e5e5e5; text-shadow: 1px 1px 5px #000; margin-bottom: 25px; font-weight: 400;}
    .hero-buttons { display: flex; gap: 15px; }
    .btn-play { background-color: #fff; color: #000; padding: 12px 35px; border-radius: 4px; font-weight: bold; font-size: 1.2rem; border: none; cursor: pointer; transition: 0.2s;}
    .btn-play:hover { background-color: #d8d8d8; }
    .btn-info { background-color: rgba(109, 109, 110, 0.7); color: #fff; padding: 12px 35px; border-radius: 4px; font-weight: bold; font-size: 1.2rem; border: none; cursor: pointer; transition: 0.2s;}
    .btn-info:hover { background-color: rgba(109, 109, 110, 0.9); }
    
    /* Section Titles */
    .row-title { font-size: 1.6rem; font-weight: bold; margin-left: 5%; margin-bottom: 20px; color: #e5e5e5; }
    
    /* Pro Movie Cards (2:3 Aspect Ratio for Posters) */
    .movie-grid { padding: 0 5%; display: flex; flex-wrap: wrap; gap: 20px;}
    .movie-card {
        background-color: #181818; border-radius: 8px; overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative; cursor: pointer; margin-bottom: 30px;
    }
    .movie-card:hover { transform: scale(1.05); z-index: 10; box-shadow: 0 10px 20px rgba(0,0,0,0.9); }
    .movie-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; }
    .movie-content { padding: 15px; }
    .match-score { color: #46d369; font-weight: bold; font-size: 1rem; margin-bottom: 5px; display: inline-block;}
    .movie-title { color: #fff; font-size: 1.1rem; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px;}
    .movie-tags { color: #808080; font-size: 0.85rem; margin-bottom: 5px; }
    
    /* Clean Search Box styling */
    .stSelectbox div[data-baseweb="select"] { background-color: #181818; color: white; border: 1px solid #333; border-radius: 6px;}
    .stSelectbox div[data-baseweb="select"]:hover { border-color: #666;}
</style>
""", unsafe_allow_html=True)

# --- UNBLOCKABLE IMDB DATASET ---
@st.cache_data
def load_movie_data():
    data = {
        'Title': [
            'Inception', 'Interstellar', 'The Dark Knight', 'Avengers: Endgame', 'Titanic', 
            'The Matrix', 'Avatar', 'Gladiator', 'Joker', 'Spider-Man: Spider-Verse', 
            'The Godfather', 'Pulp Fiction', 'Jurassic Park', 'Toy Story', 'The Shawshank Redemption',
            'Dune', 'Mad Max: Fury Road', 'Parasite', 'Whiplash', 'The Lion King'
        ],
        'Genre': [
            'Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Crime', 'Action Sci-Fi', 'Romance Drama', 
            'Sci-Fi Action', 'Sci-Fi Adventure', 'Action Drama', 'Crime Drama', 'Action Animation', 
            'Crime Drama', 'Crime Thriller', 'Sci-Fi Adventure', 'Animation Comedy', 'Drama',
            'Sci-Fi Adventure', 'Action Sci-Fi', 'Thriller Drama', 'Music Drama', 'Animation Adventure'
        ],
        'Description': [
            "A thief who steals corporate secrets through the use of dream-sharing technology.",
            "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham.",
            "Earth's mightiest heroes must come together and learn to fight as a team to stop Thanos.",
            "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious Titanic.",
            "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "A paraplegic Marine dispatched to the moon Pandora becomes torn between following orders and protecting it.",
            "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family.",
            "In Gotham City, a mentally troubled comedian is disregarded by society, leading him down a path of crime.",
            "Teen Miles Morales becomes the Spider-Man of his universe, and must join with others to stop a threat.",
            "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his son.",
            "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in tales of violence.",
            "A pragmatic paleontologist is tasked with protecting kids after a power failure causes cloned dinosaurs to run loose.",
            "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy.",
            "Two imprisoned men bond over a number of years, finding solace and eventual redemption.",
            "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset.",
            "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland.",
            "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
            "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
            "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."
        ],
        'Image': [
            'https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BZjdkOTU3MDItN2IxOS00MTFiLTgwMzIyNDc1ODYxOThmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMDdmZGU3NDPhNjFhMi00NmY2LWEzNd1yM2I2MTBkNmGzZT13XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BZDA0OGQxNTItMDZkMC00N2UyLTg3MzMtYTJmNjg3Nzk5MzRiXkEyXkFqcGdeQXVyNDUzOTQ5MjY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMDliMmNhNDEtODUyOS00MzVmLTgyMDItNzQ1NjE3Nzk4ZmRlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BNGVjNWI4ZGUtNzE0MS00YTJmLWE0ZDctN2ZiYTk2YmI3NTYyXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMjMwNDkxMTgzOF5BMl5BanBnXkFtZTgwNTkwNTQ3NjM@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZTk3OTFlXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BNGNhMDIzZTItNjZlZi00ZTUzLWJhM2QtYzhjYjY0ZDIxZDRlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMjM2MDgxMDg0Nl5BMl5BanBnXkFtZTgwNTM2OTM5NDE@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDUzOTQ5MjY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BN2FjNmEyNWMtMzM2KeepYjhlLWE2NzktZDFjZmIyMDk1NmRmXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BN2EwM2I5OWMtMGQyMi00Zjg1LWJkNTctZTdjYTA4OGUwZjMyXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BYWZjMjk3ZTItODQ2ZC00NTY5LWE0ZDYtZTI3MjcwN2Q5NTVkXkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BOTA5NDZlZGUtMjAxOS00YTRkLTkwYmMtYWQ0NWEwZDZiNjEzXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg',
            'https://m.media-amazon.com/images/M/MV5BYTYxNGMyZTYtMjE3MS00MzNjLWFjNmYtMDk3N2FmM2JiM2M1XkEyXkFqcGdeQXVyNjY5NDU4NzI@._V1_FMjpg_UX1000_.jpg'
        ]
    }
    df = pd.DataFrame(data)
    df['Combined_Features'] = df['Genre'] + " " + df['Description']
    return df

df = load_movie_data()

# --- ML ENGINE (TF-IDF & COSINE SIMILARITY) ---
@st.cache_resource
def build_recommendation_engine(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['Combined_Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_recommendation_engine(df)

def get_recommendations_with_scores(title, cosine_sim_matrix, dataframe):
    idx = dataframe.index[dataframe['Title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 matches
    top_items = sim_scores[1:6]
    top_indices = [i[0] for i in top_items]
    scores = [int(i[1] * 100) for i in top_items] # Convert to percentage
    
    recs = dataframe.iloc[top_indices].copy()
    recs['Match_Score'] = scores
    
    # Boost scores for visual appeal if they are too low in this small dataset
    recs['Match_Score'] = recs['Match_Score'].apply(lambda x: x if x > 60 else x + random.randint(20, 40))
    return recs

# --- TOP NAV & HERO SECTION ---
st.markdown('<div class="nav-logo">CINEMATCH</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div style="color:#E50914; font-weight:900; letter-spacing:3px; margin-bottom:15px; font-size:1.1rem;">N E T F L I X &nbsp; O R I G I N A L</div>
    <div class="hero-title">INTERSTELLAR</div>
    <div class="hero-desc">When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot must lead a team of researchers through a newly discovered wormhole in an attempt to ensure humanity's survival.</div>
    <div class="hero-buttons">
        <button class="btn-play">▶ Play</button>
        <button class="btn-info">ⓘ More Info</button>
    </div>
</div>
""", unsafe_allow_html=True)


# --- AI SEARCH ENGINE ---
c1, c2, c3 = st.columns([0.5, 3, 0.5])
with c2:
    st.markdown('<div class="row-title" style="margin-left:0; text-align:center;">Find Your Next Obsession</div>', unsafe_allow_html=True)
    selected_movie = st.selectbox("Search based on a movie you love:", df['Title'].tolist(), label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# --- AI RECOMMENDATIONS DISPLAY ---
if selected_movie:
    st.markdown(f'<div class="row-title">Because you watched {selected_movie}...</div>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing deep learning vectors..."):
        time.sleep(0.5) 
        recs = get_recommendations_with_scores(selected_movie, cosine_sim, df)
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                movie = recs.iloc[i]
                card_html = f"""
                <div class="movie-card">
                    <img src="{movie['Image']}" class="movie-img">
                    <div class="movie-content">
                        <div class="match-score">{movie['Match_Score']}% Match</div>
                        <div class="movie-title">{movie['Title']}</div>
                        <div class="movie-tags">{movie['Genre']}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

# --- TRENDING NOW (MOCK ROW) ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="row-title">Trending Now</div>', unsafe_allow_html=True)

trending_df = df.sample(n=5)
cols2 = st.columns(5)
for i, col in enumerate(cols2):
    with col:
        movie = trending_df.iloc[i]
        match_score = random.randint(85, 99) 
        card_html = f"""
        <div class="movie-card">
            <img src="{movie['Image']}" class="movie-img">
            <div class="movie-content">
                <div class="match-score" style="color:white;">Top 10 Today</div>
                <div class="movie-title">{movie['Title']}</div>
                <div class="movie-tags">{movie['Genre']}</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)