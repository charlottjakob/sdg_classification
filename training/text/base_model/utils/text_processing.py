import re
# from .utils import max_len
import numpy as np
import nltk
import os
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def text_preprocessing(df, lemmantization=False, stop_words=False, lower_case=False):
    # only keep letters and spaces
    df['text'] = df['text'].apply(lambda x: re.sub('[^A-Za-z\s]+', ' ', x))
    df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

    # Lower Case
    if lower_case:
        df['text'] = df['text'].apply(lambda x: x.lower())

    # Stop Words
    if stop_words:
        df['text'] = df['text'].apply(lambda x: " ".join([word for word in x.split(' ') if word not in stopwords]))

    # Lemm
    if lemmantization:
        stemmer = SnowballStemmer(language='english')
        df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split(' ')]))

    return df


def text_transfrom(text_series, embedding_model):

    # get vocabulary to be able to check if words are transfromable
    vocab = list(embedding_model.key_to_index.keys())

    # Initialize list to save words that could not be transformed
    words_not_in_vocab = []
    word_count = 0
    word_collection = set([])

    # loop through datapoints transfrom and append to transfromed
    transformed = []
    for word_list in text_series.apply(lambda x: x.split(" ")):
        vectors_datapoint = []

        # loop through the single words of the datapoint and append to vectors_datapoint
        for word in word_list:
            word_count += 1
            word_collection.add(word)
            if word in vocab:
                vectors_datapoint.append(embedding_model[word])
            else:
                words_not_in_vocab.append(word)

        # Pad datapoint with zeros or shorten it to get exactly max_len word-vectors for every datapoint
        n_words = len(vectors_datapoint)
        vector_size = len(vectors_datapoint[0])
        if n_words < 512:
            vectors_datapoint = np.concatenate((np.array(vectors_datapoint), np.zeros((512 - n_words, vector_size))))
        elif n_words > 512:
            vectors_datapoint = vectors_datapoint[:512]

        transformed.append(vectors_datapoint)

    # reshape data to have one vector per datapoint
    X_feature_space = np.array(transformed).reshape((len(text_series), -1))

    # Count occurence of words that are not transformed
    dict_words_not_transformed = Counter(words_not_in_vocab)
    amount_not_transformed = sum(dict_words_not_transformed.values())
    print('amount of unique words in dataset: ', len(word_collection))
    print('words not transformed amount: ', amount_not_transformed, ' percentage: ', '{}.3f'.format(amount_not_transformed / word_count))

    return X_feature_space, dict_words_not_transformed
