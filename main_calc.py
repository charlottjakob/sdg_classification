from training_data.build_basic_dict import combine_un_data
from training_data.clean_dict import clean_basic
from training_data.vector_embedding import word2vec
from training_data.extract_keywords import extract_keywords
from training_data.tokenize import tokenize_to_sentences


df = combine_un_data()
df = clean_basic(df)
# df = extract_keywords(df)
df = tokenize_to_sentences(df)

print('here')


import pandas as pd
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

data = df[['goal', 'text']].rename(columns={"goal":"label", "text":"text"})
 
data['label'] = '__label__' + data['label'].astype(str)

data.iloc[0:int(len(data)*0.8)].to_csv('files/flair/train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('files/flair/test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('files/flair/dev.csv', sep='\t', index = False, header = False);

# column format indicating which columns hold the text and label(s)
column_name_map = {1: "label_topic", 2: "text"}
data_folder = 'files/flair'
label_type = 'question_class'
# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder=data_folder,
                                         column_name_map=column_name_map,
                                         skip_header=True,
                                         delimiter=',',
                                         label_type =label_type,    # tab-separated files
) 

# corpus = NLPTaskDataFetcher.load_classification_corpus(Path('files/flair'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False, label_type=label_type)

trainer = ModelTrainer(classifier, corpus)
trainer.train('files/flair', max_epochs=10)







