from langdetect import detect


# def cleaning_from_testing_data_cleaning(df):
#   df['text'] = df['text'].apply(lambda x: re.sub('[^A-Za-z\s\.]+', ' ', str(x)))
#   df['text'] = df['text'].apply(lambda x: re.sub('([A-Z\s]{10,})([A-Z]{1})(?!\s)', r' \1 \2', x))
#   df['text'] = df['text'].apply(lambda x: re.sub('(?:(?<=\s)|(?<=^))([A-Z]?[a-z]+)([A-Z][a-zA-Z]+)(?=\s|$)', r' \1 \2 ', x))
#   df['text'] = df['text'].apply(lambda x: re.sub('\s(([A-Z]\s){4,})', r' ', x))
#   df['text'] = df['text'].apply(lambda x: " ".join(x.split()).strip())

#   return df


def filter_english_language(df):
    df = df[df['text'].notna()]

    # detect language and filter for english
    # df['language'] = df['text'].apply(lambda x: detect(str(x)))
    df['language'] = ''

    for i, row in df.iterrows():

        try:
            language = detect(row['text'])

        except Exception as e:
            language = "error"
            print("This row throws and error:", row['text'])

        df.at[i, 'language'] = language

    df = df[df['language'] == 'en']

    return df
