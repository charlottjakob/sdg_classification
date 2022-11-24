import pandas as pd

def encode_sdgs_multi_label(df, sdg_column='sdg', seperator=','):

    df[[str(x) for x in range(1, 18)]] = 0.
    df = df.reset_index(drop=True)

    for i, row in df[df[sdg_column].notna()].iterrows():
        for sdg in str(row[sdg_column]).split(seperator):
            sdg = str(int(sdg))  # first transform float into int and than into string  e.g. 1.0 -> 1 -> '1'
            df.at[i, sdg] = 1.

    return df


def encode_sdgs_multi_class(df, sdg_column='sdg'):

    df['label'] = df[sdg_column] - 1

    return df
