from thefuzz import process
from nltk.util import ngrams
from utils.sdg_description import sdg_short_description, sdg_long_description
import logging
import pandas as pd


logging.basicConfig(filename='log_test',
                    filemode='a',
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    # datefmt='%H:%M:%S',
                    level=logging.DEBUG)
longs = [x.replace(".","").replace(",","")  for x in  sdg_long_description.values()]


def delete_sdg_descriptions(page):
    threshold = 80
    
    text = page['page_text']

    initial_ratio = process.extractOne(text, longs)[1]
    logging.info("-"*20)
    logging.info("id: "+  str(page['_id']) +  ", page_number: "+ str(page['page_number']) + ", initial similarity: "+ str(initial_ratio))
    
    if initial_ratio >= 60:

        text_length = len(text.split())
        
        n_max = max([len(x.split()) for x in longs])
        n_min = min([len(x.split()) for x in longs])

        n_min = 6 # min amount of words that will be compared

        data = []
        for n in range(n_min, n_max + 1):  # st1_length >= m_start

            # If m=3, fs1 = 'Patient has checked', 'has checked in', 'checked in for' ...
            # If m=5, fs1 = 'Patient has checked in for', 'has checked in for abdominal', ...
            for s1 in ngrams(text.split(), n):
                word_sequence = ' '.join(s1)
                
                sdg_desc, f_ratio = process.extractOne(word_sequence, longs)
                
                # Save sub string if ratio is within threshold.
                if f_ratio >= threshold:
                    data.append([word_sequence, sdg_desc, f_ratio])

        df = pd.DataFrame(data, columns=["text_sequence", "sdg_description", "similarity_score"])


        # find sequences with max similarity score
        idx = df.groupby("sdg_description")['similarity_score'].transform(max) == df['similarity_score']
        
        sequences_with_max_values =  df[idx]
        
        # find longest sequence
        sequences_with_max_values['word_count'] = sequences_with_max_values['text_sequence'].apply(lambda x: len(x.split(" ")))
        idx = sequences_with_max_values.groupby("sdg_description")['word_count'].transform(max) == sequences_with_max_values['similarity_score']
        to_be_removed = sequences_with_max_values[idx]

        logging.info("number of sequences found: " + str(len(to_be_removed)))
        if len(to_be_removed) > 0:
            logging.info(to_be_removed)

        # delete the sequences from the text
        for sequence in to_be_removed['text_sequence']:
            text = text.replace(sequence, " ")

    return text