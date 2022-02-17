
from nltk.tokenize import sent_tokenize


def tokenize_to_sentences(df):
    df['text'] = df.text.apply(sent_tokenize)

    return df.explode('text', ignore_index=True)
