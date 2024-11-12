# Import des modules
import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Application Multi-Page", page_icon="🎥")

# Charger le DataFrame
file_path = r'D:/WCS/9_Streamlit/Projet2/data/df.csv'
df = pd.read_csv(file_path)

# Chemins des images locales
local_image_paths = {
    'background': r'D:/WCS/9_Streamlit/Projet2/image/background_cinema3.png',
    'bobine': r'D:/WCS/9_Streamlit/Projet2/image/bobine.png',
    'logo': r'D:/WCS/9_Streamlit/Projet2/image/logo.png'
}

def get_image_data(image_path):
    """Convertit une image en une chaîne base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_data):
    """Définit l'image de fond de l'application."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_data});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_title_style(bobine_data, logo_data):
    """Définit le style pour le titre avec image de fond."""
    st.markdown(
        f"""
        <style>
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            background-image: url(data:image/png;base64,{bobine_data});
            background-size: cover;
            padding: -10px;
            color: white;
            height: 250px;  
            margin: 0;  
            position: relative;
            top: -50px;  
            left: -469px;  
            width: calc(100% + 1000px);            
        }}                         
        .title-container h1 {{
            color: white;
            font-family: 'Arial', sans-serif;  
        }}        
        </style>
        <div class="title-container">
            <img src="data:image/png;base64,{logo_data}" width="100"> 
            <h1>Les Recommendations selon vos préférences ! </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Encodez les images en base64
background_image_data = get_image_data(local_image_paths['background'])
bobine_image_data = get_image_data(local_image_paths['bobine'])
logo_image_data = get_image_data(local_image_paths['logo'])

# Affichage de l'image de fond principale
set_background(background_image_data)

# Affichage du titre avec l'image de fond spécifique
set_title_style(bobine_image_data, logo_image_data)

# Prétraitement des données
columns_to_exclude = ['popularity']
X = df.select_dtypes(include='number').drop(columns=columns_to_exclude)
X_scaled = RobustScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Choix des colonnes catégorielles
actors = X_scaled.loc[:, X_scaled.columns.str.contains('Actor')]
genre = X_scaled.loc[:, X_scaled.columns.str.contains('Genre')]
X_scaled[actors.columns] = 2000 * X_scaled[actors.columns] / len(actors.columns)
X_scaled[genre.columns] = 2500 * X_scaled[genre.columns] / len(genre.columns)

# Séparer les données en train et test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = X_scaled.loc[train_df.index]
X_test = X_scaled.loc[test_df.index]

# Fonction de recommandation basée sur la similarité cosinus
def recommend_movies(df, X_scaled, user_choice1, user_choice2, genres, actors):
    # Initialiser les DataFrames pour les choix de l'utilisateur
    vecteur_choice1 = pd.DataFrame()
    agg_films = pd.DataFrame()

    if user_choice1:
        index_choice1 = np.array(df.loc[df['original_title'] == user_choice1].index)
        if index_choice1.size > 0:
            vecteur_choice1 = X_scaled.loc[X_scaled.index.isin(index_choice1)].mean().to_frame().T

    if user_choice2:
        actor_col_name = 'Actor_' + user_choice2
        if actor_col_name in df.columns:
            index_choice2 = np.array(df.loc[df[actor_col_name] == 1].index)
            if index_choice2.size > 0:
                agg_films = X_scaled.loc[X_scaled.index.isin(index_choice2)].mean(axis=0).to_frame().T

    # Filtrer par genres
    if genres:
        genre_columns = [f'Genre_{genre}' for genre in genres if f'Genre_{genre}' in X_scaled.columns]
        if genre_columns:
            genre_filter = X_scaled[genre_columns].sum(axis=1) > 0
            X_scaled = X_scaled[genre_filter]
            df = df[genre_filter]
        else:
            return pd.DataFrame()  # Aucun genre correspondant trouvé

    # Filtrer par acteurs
    if actors:
        actor_columns = [f'Actor_{actor}' for actor in actors if f'Actor_{actor}' in X_scaled.columns]
        if actor_columns:
            actor_filter = X_scaled[actor_columns].sum(axis=1) > 0
            X_scaled = X_scaled[actor_filter]
            df = df[actor_filter]
        else:
            return pd.DataFrame()  # Aucun acteur correspondant trouvé

    if vecteur_choice1.empty and agg_films.empty:
        return pd.DataFrame()  # Aucun choix valable pour la recommandation

    # Assurer que X_scaled et vecteur_choice1 ne sont pas vides avant le calcul des similarités
    if not vecteur_choice1.empty:
        similarites = cosine_similarity(X_scaled, vecteur_choice1)
        df['similarity_with_choice1'] = similarites
    else:
        df['similarity_with_choice1'] = 0

    if not agg_films.empty:
        similarites_actor = cosine_similarity(X_scaled, agg_films)
        df['similarity_with_choice2'] = similarites_actor
    else:
        df['similarity_with_choice2'] = 0

    # Filtrer les films pour avoir ceux les plus similaires aux choix
    recommendations = df[['original_title', 'poster_path', 'release_year', 'runtime_minutes', 'overview', 'similarity_with_choice1', 'similarity_with_choice2']]
    
    # Retirer les duplications
    recommendations = recommendations.drop_duplicates(subset=['original_title'])
    
    # Trier les recommandations
    recommendations = recommendations.sort_values(by=['similarity_with_choice1', 'similarity_with_choice2'], ascending=False)
    
    return recommendations.head(4)  # Retourne les 4 meilleures recommandations

# Sidebar pour les filtres
# with st.sidebar:
#     search_title = st.text_input('Saisir le titre de votre film:')
#     search_actors = st.text_input('Saisir le nom d’un acteur:')

#     # Récupérer les genres disponibles à partir du DataFrame
#     genres_list = sorted(df.filter(like='Genre').columns.str.replace('Genre_', '').tolist())
    
#     # Ajouter un sélecteur pour les genres
#     selected_genres = st.multiselect('Sélectionner un ou plusieurs genres:', options=genres_list)

#     submitted = st.button("🎬")

with st.sidebar:
    search_title = st.text_input('Saisir le titre de votre film:')
    # search_actors = st.text_input('Saisir le nom d’un acteur:')

    # Récupérer les genres disponibles à partir du DataFrame
    genres_list = sorted(df.filter(like='Genre').columns.str.replace('Genre_', '').tolist())
    
    # Ajouter un sélecteur pour les genres
    selected_genres = st.multiselect('Sélectionner un ou plusieurs genres:', options=genres_list)

    # Liste des acteurs sélectionnés dans une case à cocher par exemple
    actors_list = sorted(df.filter(like='Actor').columns.str.replace('Actor_', '').tolist())
    selected_actors = st.multiselect('Sélectionner un ou plusieurs acteurs:', options=actors_list)

    submitted = st.button("🎬")

# Lire l'image de remplacement pour les affichages NaN
def get_image_data(image_path):
    """Convertit une image en une chaîne base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Chemin de l'image de remplacement
default_image_path = r'D:/WCS/9_Streamlit/Projet2/image/logo.png'
default_image_data = get_image_data(default_image_path)

# Affichage des recommandations dans la section principale après soumission
# if submitted:
#     recommendations = recommend_movies(df, X_scaled, search_title, search_actors, selected_genres)
#     if recommendations.empty:
#         st.write("Aucune recommandation trouvée.")
if submitted:
    recommendations = recommend_movies(df, X_scaled, search_title, search_actors, selected_genres, selected_actors)
    if recommendations.empty:
        st.write("Aucune recommandation trouvée.")
    else:
        st.markdown("<h2 style='text-align: center; color: white;'>Films recommandés</h2>", unsafe_allow_html=True)
        
        # Affichage des films recommandés
        url = 'https://image.tmdb.org/t/p/original'
        rows = st.columns(2)

        for i in range(len(recommendations)):
            with rows[i % 2].container():
                col1, col2 = st.columns(2)
                poster_path = recommendations.iloc[i]['poster_path']
                
                # Assurer que poster_path est une chaîne de caractères valide et non NaN
                if pd.notna(poster_path) and isinstance(poster_path, str):
                    col1.image(url + poster_path, use_column_width='auto')
                else:
                    # Afficher l'image de remplacement depuis les données base64
                    col1.image(f"data:image/png;base64,{default_image_data}", use_column_width='auto')
                
                with col2:
                    st.markdown(f"**Titre Original :** {recommendations.iloc[i]['original_title']}")
                    st.markdown(f"**Titre :** {recommendations.iloc[i]['title']}")
                    st.markdown(f"**Année de sortie :** {recommendations.iloc[i]['release_year']}")
                    st.markdown(f"**Durée :** {recommendations.iloc[i]['runtime_minutes']} min")
                    st.markdown(f"**Résumé :** {recommendations.iloc[i]['overview']}")
                    st.markdown("---")

if not submitted:
    st.markdown("<h2 style='text-align: center; color: white;'>Films aléatoires</h2>", unsafe_allow_html=True)
    
    # Suppression des doublons dans le DataFrame
    df_unique = df.drop_duplicates(subset=['original_title', 'poster_path', 'release_year', 'runtime_minutes', 'overview'])
    
    # Sélection aléatoire de 4 films
    random_films = df_unique.sample(n=4)  # random_state assure que l'échantillon est reproductible

    # Affichage des films
    url = 'https://image.tmdb.org/t/p/original'
    rows = st.columns(2)

    for i in range(len(random_films)):
        with rows[i % 2].container():
            col1, col2 = st.columns(2)
            poster_path = random_films.iloc[i]['poster_path']
            
            # Assurer que poster_path est une chaîne de caractères valide et non NaN
            if pd.notna(poster_path) and isinstance(poster_path, str):
                col1.image(url + poster_path, use_column_width='auto')
            else:
                # Afficher l'image de remplacement depuis les données base64
                col1.image(f"data:image/png;base64,{default_image_data}", use_column_width='auto')
            
            with col2:
                st.markdown(f"**Titre Original :** {random_films.iloc[i]['original_title']}")
                st.markdown(f"**Titre :** {random_films.iloc[i]['title']}")
                st.markdown(f"**Année de sortie :** {random_films.iloc[i]['release_year']}")
                st.markdown(f"**Durée :** {random_films.iloc[i]['runtime_minutes']} min")
                st.markdown(f"**Résumé :** {random_films.iloc[i]['overview']}")
                st.markdown("---")
                
                
