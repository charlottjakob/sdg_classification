"""Extract full text from single page pdfs."""

# basic
import pandas as pd
import os

# text extraction
import PyPDF2
os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')
df = pd.read_csv('data/reports_prediction.csv')  # final_pages.csv

# filter for pages with sdgs and in english
df = df[df['predictions'].notna()]  #df['sdgs'].notna() & (df['information'] == 'yes') & (df['language'] == 'english')

df = df[['file_name', 'page', 'predictions']]
df.reset_index(inplace=True, drop=True)

len_df = len(df)

for i, page in df.iterrows():

  try:
    file_name = page['file_name']
    page_number = int(page['page'])
    document_path = f'data/sustainability_reports_500_1500/{file_name}'

    # creating a pdf file object
    pdf_file_obj = open(document_path, 'rb')

    # creating a pdf reader object
    pdf = PyPDF2.PdfFileReader(pdf_file_obj)

    pdf_page = pdf.getPage(page_number)

    text = pdf_page.extractText().replace("\r", " ").replace("\n", " ")

    pdf_file_obj.close()

    df.at[i, 'text_extraction_success'] = 'yes'
    df.at[i, 'text'] = text

    print('done: ', i, " / ", len_df)

  except Exception as e:

    df.at[i, 'text_extraction_success'] = 'no'
    df.at[i, 'error'] = e
    print('failed: ', document_path)

df.to_csv('data/sr_data.csv')
