"""Build basic dictionary."""
import pandas as pd
import os
from training_data.utils.sdg_description import sdg_short_description


def load_un_data():
    os.chdir('/Users/charlottjakob/Documents/ecoworld/sdg-classification')

    un_data = pd.read_csv("files/un_data.csv")

    # prepare un_data for concatination

    # take targets only once
    un_data_target_text = un_data[un_data['target_text'].notna()][['goal', 'target_text']].drop_duplicates('target_text').rename(columns={'target_text': 'text'})
    un_data_target_text['text'] = un_data_target_text['text'] + "."
    # take indicators
    un_data_indicator_text = un_data[un_data['indicator_text'].notna()][['goal', 'indicator_text']].rename(columns={'indicator_text': 'text'})
    un_data_indicator_text['text'] = un_data_indicator_text['text'] + "."
    # take infos
    un_data_info_text = un_data[un_data['info_text'].notna()][['goal', 'info_text']].rename(columns={'info_text': 'text'})
    # take related_topic_texts
    un_data_related_topic_text = un_data[un_data['related_topic_text'].notna()][['goal', 'related_topic_text']].rename(columns={'related_topic_text': 'text'})

    goal_text_df_splitted = pd.concat([un_data_target_text, un_data_indicator_text, un_data_info_text, un_data_related_topic_text])

    return goal_text_df_splitted


def combine_un_data():
    """Load data and combine texts to dict.

    Returns:
        dict with key: SDG, value:text
    """
    goal_text_df_splitted = load_un_data()

    goal_text_df_splitted = goal_text_df_splitted[goal_text_df_splitted['text'].notna()]
    goal_text_df_splitted['text'] = goal_text_df_splitted['text'].apply(lambda x: str(x))
    goal_text_df_splitted['goal'] = goal_text_df_splitted['goal'].apply(lambda x: int(x))
    goal_text_df = goal_text_df_splitted.groupby(['goal'])['text'].apply(lambda x: ' '.join(x)).reset_index()
    goal_text_df['short_description'] = goal_text_df['goal'].apply(lambda x: sdg_short_description[str(x)])
    # goal_text_df.T.to_dict()
    return goal_text_df


def combine_un_and_scholar_data():
    """Load data and combine texts to dict.

    Returns:
        dict with key: SDG, value:text
    """
    os.chdir('/Users/charlottjakob/Documents/ecoworld/sdg-classification')

    # load an prepare scholar_data
    scholar_data = pd.read_csv("files/scholar_data.csv")
    scholar_data_pdf_text = scholar_data[['goal', 'pdf_text']].rename(columns={'pdf_text': 'text'})

    # load un_data
    un_data = load_un_data()

    # concat scholar_data and un_data
    goal_text_df_splitted = pd.concat([scholar_data_pdf_text, un_data])

    goal_text_df_splitted = goal_text_df_splitted[goal_text_df_splitted['text'].notna()]
    goal_text_df_splitted['text'] = goal_text_df_splitted['text'].apply(lambda x: str(x))
    goal_text_df = goal_text_df_splitted.groupby('goal')['text'].apply(lambda x: ' '.join(x)).reset_index()
    goal_text_df['short_description'] = goal_text_df_splitted['goal'].apply(lambda x: sdg_short_description[str(x)])

    return goal_text_df
