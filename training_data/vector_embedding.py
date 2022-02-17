"""Functions for vector embedding."""
import gensim
from training_data.tokenize import tokenize_to_sentences

def word2vec(df):
    """Train word2vec with gensim.

    Args:
        df: DataFrame with columns "SDG", "text"
    """
    # split text to sentences
    df = tokenize_to_sentences(df)

    # split sentenes to words
    df.text = df.text.apply(gensim.utils.simple_preprocess)

    # train model
    model = gensim.models.Word2Vec(
        window=4,
        min_count=3,
    )
    model.build_vocab(df.text, progress_per=1000)
    model.train(df.text, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('word2vec.model')
