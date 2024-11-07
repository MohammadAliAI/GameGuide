import streamlit as st
import pandas as pd

@st.cache
def load_and_preprocess_data(file_path: str):
    df = pd.read_csv(file_path)
    
    # Display column names for debugging purposes
    st.write("Column names in the dataset:", df.columns.tolist())
    
    # Adjust column names based on the actual data
    df = df[['user_id', 'game_name', 'hours']].dropna()

    # Create unique user and game indexes
    users = df['user_id'].unique()
    games = df['game_name'].unique()

    user_cat = pd.CategoricalDtype(categories=sorted(users), ordered=True)
    game_cat = pd.CategoricalDtype(categories=sorted(games), ordered=True)

    user_idx = df['user_id'].astype(user_cat).cat.codes
    game_idx = df['game_name'].astype(game_cat).cat.codes

    df['UserIndex'] = user_idx
    df['GameIndex'] = game_idx

    return df, user_idx, game_idx
