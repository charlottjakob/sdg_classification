"""Build basic dictionary."""
import pandas as pd
import os
from training.utils.sdg_description import sdg_short_description


def combine_training_data(nosdg_data_included):
    """Load data and combine texts to dict.

    Returns:
        dict with key: SDG, value:text
    """
    os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')

    # load an prepare scholar_data
    scholar_data = pd.read_csv("data/scholar_data.csv")
    scholar_data = scholar_data[['sdg', 'abstract']].rename(columns={'abstract': 'text'})

    # load un_data
    un_data = load_un_data()

    if nosdg_data_included:
        wiki_data = pd.read_csv("data/wikipedia_data.csv")
        wiki_data['text'] = wiki_data['text'].apply(lambda x: x.split("|||"))
        wiki_data = wiki_data.explode('text')
        wiki_data['sdg'] = 0
        wiki_data = wiki_data.drop('term', axis=1)
        df = pd.concat([scholar_data, un_data, wiki_data])

    else:
        # concat scholar_data and un_data
        df = pd.concat([scholar_data, un_data])

    df = df[df['text'].notna()]
    df['text'] = df['text'].apply(lambda x: str(x))
    # df = df_splitted.groupby('sdg')['text'].apply(lambda x: ' '.join(x)).reset_index()

    return df


def combine_un_data():
    """Load data and combine texts to dict.

    Returns:
        dict with key: SDG, value:text
    """
    df_splitted = load_un_data()

    df_splitted = df_splitted[df_splitted['text'].notna()]
    df_splitted['text'] = df_splitted['text'].apply(lambda x: str(x))
    df_splitted['sdg'] = df_splitted['sdg'].apply(lambda x: int(x))
    # df = df_splitted.groupby(['sdg'])['text'].apply(lambda x: ' '.join(x)).reset_index()
    return df_splitted


def load_un_data():
    os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')

    un_data = pd.read_csv("data/un_data.csv")

    # prepare un_data for concatination

    # take targets only once
    un_data_target_text = un_data[un_data['target_text'].notna()][['sdg', 'target_text']].drop_duplicates('target_text').rename(columns={'target_text': 'text'})
    un_data_target_text['text'] = un_data_target_text['text'] + "."
    # take indicators
    un_data_indicator_text = un_data[un_data['indicator_text'].notna()][['sdg', 'indicator_text']].drop_duplicates('indicator_text').rename(columns={'indicator_text': 'text'})
    un_data_indicator_text['text'] = un_data_indicator_text['text'] + "."
    # take infos
    un_data_info_text = un_data[un_data['info_text'].notna()][['sdg', 'info_text']].rename(columns={'info_text': 'text'})
    # take related_topic_texts
    un_data_related_topic_text = un_data[un_data['related_topic_text'].notna()][['sdg', 'related_topic_text']].rename(columns={'related_topic_text': 'text'})

    goal_text_df_splitted = pd.concat([un_data_target_text, un_data_indicator_text, un_data_info_text, un_data_related_topic_text])

    return goal_text_df_splitted
