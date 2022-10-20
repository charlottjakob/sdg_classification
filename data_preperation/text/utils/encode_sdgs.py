
def encode_sdgs_multi_label(df, sdg_column='sdg', seperator=','):

    df[[str(x) for x in range(0, 18, 1)]] = 0.

    for i, row in df.iterrows():
        for sdg in str(row[sdg_column]).split(seperator):
            sdg = int(sdg)
            df.at[i, str(sdg)] = 1.

    return df


def encode_sdgs_multi_class(df, sdg_column='sdg'):

    df['label'] = df[sdg_column] - 1

    return df
