import pandas as pd
from utils.tokenize import tokenize_to_sentences


def company_to_cop_sentences(company_name="Beauty Industry Group"):

    df = pd.read_csv('files/cop_data.csv')

    df = df.sample(1)
    df = tokenize_to_sentences(df, 'text')
    df = df.drop_duplicates('text')
    df.to_csv('files/cop_example_tokenized.csv')
