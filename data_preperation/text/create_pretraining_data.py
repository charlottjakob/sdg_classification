# importing required modules
import PyPDF2
import pandas as pd
import random
from text_cleaning import base_cleaning
import re

companies = pd.read_csv('data/reports_approval_manual.csv')
companies = companies[companies['approved_final'] == 'yes']
companies = companies[['document_file_name']]

text_domain = pd.DataFrame()
for i, company in companies[:500].iterrows():

    try:
        document_path = 'data/sustainability_reports_500_1500/' + str(company['document_file_name'])

        # creating a pdf file object
        pdf_file_obj = open(document_path, 'rb')

        # creating a pdf reader object
        pdf = PyPDF2.PdfFileReader(pdf_file_obj)


        # change to just randomly take 20 -> we don't connection for BERT
        total_page = pdf.numPages
        if total_page > 22:
            page_start = random.sample(range(2, total_page - 19), 1)[0]
            page_end = page_start + 20
        else:
            page_start = 2
            page_start = total_page

        text = ""

        for current_page in range(page_start, page_end):
            pdf_page = pdf.getPage(current_page)

            try:
                text = pdf_page.extractText().replace("\r", "").replace("\n", "")
                text_domain = text_domain.append({'document': company['document_file_name'], 'page': current_page, 'text': text}, ignore_index=True)

            except Exception as e:
                print('E: ', e)
                print('FAILED number: ', i, ' page: ', current_page)

        pdf_file_obj.close()
    except Exception as e:
        print('E: ', e)
        print('document: ', document_path)


text_domain = text_domain[text_domain['text'].notna()]
text_domain = base_cleaning(text_domain)

text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('[^A-Za-z\s\.]+', ' ', x))  # \'
text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('(?<=[\.\s])[A-Ya-z](?=[\.\s])', ' ', x))
text_domain['text'] = text_domain['text'].apply(lambda x: re.sub('(?<![a-z])\.', ' ', x))
text_domain['text'] = text_domain['text'].apply(lambda x: " ".join(x.split()))

text_domain = text_domain[text_domain['text'].notna()]

text_domain.to_csv('data/text_domain.csv')
