import streamlit as st
import pandas as pd
from recommender import Recommender
from utils import load_and_preprocess_data
from typing import Union, List, Dict, Any

SIDEBAR_DESCRIPTION = """
# Game Recommender System

## What is it?
A recommender system suggests new games to a user based on their play history and similar users' play history.

## How does it work?
It identifies similar users and suggests games they have played but the current user hasn't.

## Data Preparation
For each user, the system computes the total hours spent on each game. This metric is used to calculate similarity.
"""


def load_and_preprocess_data(file_path: str):
    df = pd.read_csv(file_path)
    df = df[['user_id', 'game_name', 'hours']].dropna()
    users = df['user_id'].unique()
    games = df['game_name'].unique()

    user_cat = pd.CategoricalDtype(categories=sorted(users), ordered=True)
    game_cat = pd.CategoricalDtype(categories=sorted(games), ordered=True)

    user_idx = df['user_id'].astype(user_cat).cat.codes
    game_idx = df['game_name'].astype(game_cat).cat.codes

    df['UserIndex'] = user_idx
    df['GameIndex'] = game_idx

    return df, user_idx, game_idx


def create_and_fit_recommender(
    model_name: str,
    values: Union[pd.DataFrame, "np.ndarray"],
    users: Union[pd.DataFrame, "np.ndarray"],
    products: Union[pd.DataFrame, "np.ndarray"],
) -> Recommender:
    recommender = Recommender(
        values,
        users,
        products,
    )

    recommender.create_and_fit(
        model_name,
        model_params=dict(
            factors=50,
            regularization=0.01,
            iterations=20,
            random_state=42,
        ),
    )
    return recommender

def explain_recommendation(recommender: Recommender, user_id: int, suggestions: List[int], df: pd.DataFrame):
    output = []

    n_recommended = len(suggestions)
    for suggestion in suggestions:
        explained = recommender.explain_recommendation(
            user_id, suggestion, n_recommended
        )

        suggested_items_id = [id[0] for id in explained]

        suggested_description = df.loc[df.GameIndex == suggestion]['game_name'].unique()[0]
        similar_items_description = df.loc[df["GameIndex"].isin(suggested_items_id)]['game_name'].unique()

        output.append(f"The game **{suggested_description}** has been suggested because it is similar to the following games played by the user:")
        for description in similar_items_description:
            output.append(f"- {description}")

    with st.expander("See why the model recommended these games"):
        st.write("\n".join(output))

    st.write("------")

def print_suggestions(suggestions: List[int], df: pd.DataFrame):
    similar_items_description = df.loc[df["GameIndex"].isin(suggestions)]['game_name'].unique()

    output = ["The model suggests the following games:"]
    for description in similar_items_description:
        output.append(f"- {description}")

    st.write("\n".join(output))

def display_user_char(user: int, data: pd.DataFrame):
    subset = data[data.UserIndex == user]

    st.write(
        "The user {} played {} distinct games. Here is the play history: ".format(
            user, subset["game_name"].nunique()
        )
    )
    st.dataframe(
        subset.sort_values("hours").drop(
            ["user_id", "game_name", "hours"],
            axis=1,
        )
    )
    st.write("-----")

def main():
    st.sidebar.markdown(SIDEBAR_DESCRIPTION)

    # Load and process data
    data, users, games = load_and_preprocess_data("steam_user_train.csv")

    # Add a section to visualize the data
    st.markdown("## Data Visualization")
    if st.checkbox("Show entire dataset"):
        st.dataframe(data)  # Display the entire dataframe
    else:
        st.dataframe(data.head(100))  # Display the first 100 rows

    # Create and fit the recommender model
    recommender = create_and_fit_recommender(
        "als",
        data["hours"],
        users,
        games,
    )

    st.markdown("## GameGuide")
    with st.form("recommend"):
        user = st.selectbox("Select a user to get their game recommendations", users.unique())
        items_to_recommend = st.slider("How many games to recommend?", 1, 10, 5)
        submitted = st.form_submit_button("Recommend!")
        if submitted:
            display_user_char(user, data)
            suggestions_and_score = recommender.recommend_products(user, items_to_recommend)
            print_suggestions(suggestions_and_score[0], data)
            explain_recommendation(recommender, user, suggestions_and_score[0], data)

main()
