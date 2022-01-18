import pandas as pd
import json
import os


def build_basic_dict():
    
    os.chdir('/Users/charlottjakob/Documents/ecoworld/sdg-classification')

    scholar_data = pd.read_csv("files/scholar_data.csv")
    un_data = pd.read_csv("files/un_data.csv")

    # prepare scholar_data for concatination
    scholar_data_pdf_text = scholar_data[['goal','pdf_text']].rename(columns={'pdf_text': 'text'})

    # prepare un_data for concatination
    # take targets only once
    un_data_target_text = un_data[['goal','target_text']].drop_duplicates('target_text').rename(columns={'target_text': 'text'})
    # take indicators
    un_data_indicator_text = un_data[['goal','indicator_text']].rename(columns={'indicator_text': 'text'})
    # take infos 
    un_data_info_text = un_data[['goal','info_text']].rename(columns={'info_text': 'text'})
    # take related_topic_texts
    un_data_related_topic_text = un_data[['goal','related_topic_text']].rename(columns={'related_topic_text': 'text'})

    goal_text_df_splitted = pd.concat([scholar_data_pdf_text, un_data_target_text, un_data_indicator_text, un_data_info_text, un_data_related_topic_text])
    
    goal_text_df_splitted = goal_text_df_splitted[goal_text_df_splitted['text'].notna()]
    goal_text_df_splitted['text'] = goal_text_df_splitted['text'].apply(lambda x: str(x))
    goal_text_df = goal_text_df_splitted.groupby('goal')['text'].apply(lambda x: ''.join(x)).reset_index()

    # goal_text_df.T.to_dict()
    return goal_text_df



