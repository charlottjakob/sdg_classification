

def clean_training_data(df):

    # starts with uppter letter
    # df = df[df['text'].apply(lambda x: str(x)[0].isupper())]
    df.reset_index(inplace=True, drop=True)

    df = df.drop_duplicates('text')
    df['word_count'] = df['text'].apply(lambda x: len(x.split(" ")))
    df = df[df['word_count'] > 10]
    df = df.drop('word_count', axis=1)

    # ends with the last period
    df = df[df['text'].str.match(r'.*\.')]
    df['text'] = df['text'].str.extract(r'(.*\.)')

    # delete brackets
    df['text'] = df['text'].str.replace(r'\(.*\)', '')

    # delete brackets
    df['text'] = df['text'].apply(lambda x: " ".join(str(x).replace('\n', '').replace('\r', '').split()))

    return df
