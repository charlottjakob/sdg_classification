"""This Script creates builds the data for Pre-Training by extracting text from random pages of already scraped sustainability reports."""
# locals
from text_cleaning import base_cleaning

# basics
import pandas as pd
import random
import re

# pdf handling
import PyPDF2

AMOUNT_REPORTS = 500
AMOUNT_PAGES_PER_REPORT = 20

# get the reports file_names
companies = pd.read_csv('data/reports_approval_manual.csv')

# filter for approved reports to get the ones that are extractable and in english
companies = companies[companies['approved_final'] == 'yes'][['document_file_name']]

# initialize empty list to be filled with texts
text_domain = []

# for each report extract text and save in DataFrame
for i, company in companies[:AMOUNT_REPORTS].iterrows():

    try:
        # build path according to file_name
        document_path = 'data/sustainability_reports_500_1500/' + str(company['document_file_name'])

        # creating a pdf file object
        pdf_file_obj = open(document_path, 'rb')

        # creating a pdf reader object
        pdf = PyPDF2.PdfFileReader(pdf_file_obj)

        # change to just randomly take 20 -> we don't connection for BERT
        total_page = pdf.numPages

        # if report has more pages than the wanted pages + 2 (because the first 2 pages usually have less text)
        if total_page > AMOUNT_PAGES_PER_REPORT + 2:
            # choose random pages
            pages_to_be_extracted = random.sample(range(2, total_page), AMOUNT_PAGES_PER_REPORT)

        # else take all pages from 2 on
        else:
            pages_to_be_extracted = range(2, total_page)

        # for each page extract text and append it to list
        for current_page in pages_to_be_extracted:

            # extract text from page
            pdf_page = pdf.getPage(current_page)

            try:
                # delete \r and \n and append to list with respective file_name and page
                text = pdf_page.extractText().replace("\r", "").replace("\n", "")
                text_domain.append({'document': company['document_file_name'], 'page': current_page, 'text': text})

            except Exception as e:
                print('E: ', e)
                print('FAILED number: ', i, ' page: ', current_page)

        # close current report before opening the next one
        pdf_file_obj.close()

    except Exception as e:
        print('E: ', e)
        print('document: ', document_path)

# Transform list to dataframe for better handling
text_domain = pd.DataFrame(text_domain)

# filter for pages that include characters
text_domain = text_domain[text_domain['text'].notna()]

# perform basic cleaning as done for fine-tuning data
text_domain = base_cleaning(text_domain)

# delete special characters
text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('[^A-Za-z\s\.]+', ' ', x))

# Fix placements of periods
text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('(?<=[\.\s])[A-Ya-z](?=[\.\s])', ' ', x))
text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('(?<![a-z])\.', ' ', x))

# Delete double spaces
text_domain['text'] = text_domain['text'].apply(lambda x: " ".join(x.split()))

# filter for strings that still include characters
text_domain = text_domain[text_domain['text'].notna()]

# save data
text_domain.to_csv('data/text_domain.csv')
