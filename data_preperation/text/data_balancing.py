# basics
import numpy as np
import random

# specific
import translators as ts

CLASS_NAMES = [str(number) for number in np.arange(1, 18)]


def balance_with_ratio(df, ratios):
    df_os = df.copy()

    ratio_current = get_current_ratio(df)
    df['original'] = 1

    # add original to ratio list which represants original dataset without balancing
    ratio_float_dict = {0: 'original'}
    ratio_col_dict = {0: 'original'}
    for i, ratio in enumerate(ratios):
        ratio_float_dict[i + 1] = ratio
        ratio_col_dict[i + 1] = 'ratio_{0:.2f}'.format(ratio)

    for r in range(1, len(ratios) + 1):

        # set float and column name of current ratio
        ratio_float = ratio_float_dict[r]
        ratio_col = ratio_col_dict[r]
        print('start : ', ratio_col)

        # set samples of the previous ratio as starting point for the current ratio
        df[ratio_col] = df[ratio_col_dict[r - 1]]

        # initialize variable to be set True if balancing isn't improving the ratio anymore
        stop_balancing = False
        while ratio_current <= ratio_float and stop_balancing is False:
            # undersample
            sdg_max = df[df[ratio_col] == 1][CLASS_NAMES].sum(axis=0).idxmax()

            # find a sample with the smallest amount of positive labels in target to avoid deleting minority classes
            sdg_counts_including_sdg_max = df[(df[ratio_col] == 1) & (df[sdg_max] == 1)][CLASS_NAMES].sum(axis=1)
            idx = sdg_counts_including_sdg_max.idxmin()

            # mark sample to be not in ratio
            # df = df.drop(index=idx)
            df.at[idx, ratio_col] = 0

            # oversample 2x
            for i in range(2):

                # get sample of minority class
                sdg_min = df[df[ratio_col] == 1][CLASS_NAMES].sum(axis=0).idxmin()
                sample = df_os[(df_os[sdg_min] == 1)].sample()

                # get random language for translation
                lang_foreign = random.sample(['de', 'fr', 'es', 'it', 'pt', 'el'], 1)[0]

                try:
                    # translate in foreign language and back
                    text_german = ts.google(str(sample['text'].item()), 'en', lang_foreign)
                    sample['text'] = ts.google(text_german, lang_foreign, 'en')

                    # mark sample belonging to current ratio
                    sample[ratio_col] = 1

                    # add transformed sample to df
                    df = df.append(sample, ignore_index=True)

                except Exception as e:
                    print('failed, len: ', len(sample['text'].item()), ' lang: ', lang_foreign)
                    print('Error: ', e)

            ratio_current = get_current_ratio(df[df[ratio_col] == 1])

        #     # update current ratio
        #     ratio_new = get_current_ratio(df[df[ratio_col] == 1])
        #     if ratio_new > ratio_current:
        #         ratio_current = ratio_new
        #     else:
        #         stop_balancing = True
        #         df.rename(columns={ratio_col: 'ratio_{0:.2f}'.format(ratio_current)})
        #         print('Stopped with highest possible ratio: ', '{0:.2f}'.format(ratio_current))

        # if stop_balancing is True:
        #     # stop balancing
        #     break

    return df.fillna(0)


def get_current_ratio(df):

    # find amount of datapoints in majority class
    sdg_max = df[CLASS_NAMES].sum(axis=0).idxmax()
    sdg_max_n = len(df[df[sdg_max] == 1.0])

    # find amount of datapoints in minority class
    sdg_min = df[CLASS_NAMES].sum(axis=0).idxmin()
    sdg_min_n = len(df[df[sdg_min] == 1.0])

    # return ratio
    return sdg_min_n / sdg_max_n
