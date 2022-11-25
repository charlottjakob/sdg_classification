

def base_cleaning(df):

    # Starts with upper letter and ends with the last period
    df = df[df['text'].str.match(r'[A-Z].*\.')]
    df['text'] = df['text'].str.extract(r'([A-Z].*\.)')

    # delete brackets
    df['text'] = df['text'].str.replace(r'\(.*\)', '')

    # delete \n \r
    df['text'] = df['text'].apply(lambda x: " ".join(str(x).replace('\n', '').replace('\r', '').split()))

    df = df[~df['text'].str.contains('Source:')]

    df['word_count'] = df['text'].apply(lambda x: len(x.split(" ")))
    df = df[df['word_count'] > 10]

    df = df.drop_duplicates('text')

    return df
