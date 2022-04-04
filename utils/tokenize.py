"""Tokenization functions to split data in processable portions."""
from nltk.tokenize import sent_tokenize


def tokenize_to_sentences(df, text_column='text'):
    """Tokenize column text into sentences and expolde to have label per sentence.

    Args:
        df: DataFrame to be processed
        text_column: column of df which should be tokenized

    Returns:
        df with sentence per row
    """
    # tokenize
    df[text_column] = df[text_column].astype(str).apply(sent_tokenize)

    # explode
    return df.explode(text_column, ignore_index=True)
