import pandas as pd


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


goal_text_df_splitted = pd.concat([scholar_data_pdf_text, un_data_target_text, un_data_indicator_text, un_data_info_text])

goal_text_df = goal_text_df_splitted.groupby('goal')['text'].transform(lambda x: ''.join(str(x)))

goal_text_dict = goal_text_df.T.to_dict()
print(goal_text_dict)