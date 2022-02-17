"""Extractions."""
from keybert import KeyBERT


def extract_keywords(df):
    """Bla.

    Args:
        df: DataFrame with columns "SDG", "text"

    Returns:
        df with additional column "summarized"
    """
    model = KeyBERT()

    # for i, row in df.iterrows():
    #     keywords = model.extract_keywords(row['text'])
    #     keywords = [keyword[1] for keyword in keywords]
    #     df.at[i, 'column'] = keywords

    df['keywords'] = df.text.apply(model.extract_keywords)

    return df
