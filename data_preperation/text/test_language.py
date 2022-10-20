"""Extract text from first amount_pages pages from amount_reports reports, test it on lengh and language."""

# basic
import pandas as pd
import os

# text extraction
import PyPDF2

# language detection
from langdetect import detect

# set directory because files are not subordinated
os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')

# define parameters
amount_pages = 3
# amount_reports = 1500

def filter_reports_regarding_extractability_and_language(amount_pages=3):

    # load file_names from csv
    df_reports = pd.read_csv('data/un_global_compact_data_500_1500.csv')[1900:]  # amount_reports

    for i, report in df_reports.iterrows():

        # get path to document
        file_name = report['document_file_name']
        document_path = f'data/sustainability_reports_500_1500/{ file_name }'

        try:
            # creating a pdf file object and pdf reader
            pdf_file_obj = open(document_path, 'rb')
            pdf = PyPDF2.PdfFileReader(pdf_file_obj, strict=False)

            total_pages = int(pdf.numPages)

            # instantiate string to append the text from pages
            text = ""

            # loop through pages and extract text
            for page_number in range(amount_pages):

                pdf_page = pdf.getPage(page_number)
                try:
                    # extract text from page and append string
                    text = text + pdf_page.extractText().replace("\r", "").replace("\n", "")

                except Exception as e:
                    print(e)
                    print('text extraction failed: ' + file_name + ' page: ', page_number)

            # add amount pages to frame
            df_reports.at[i, 'total_pages'] = total_pages

            # add text to dataframe
            df_reports.at[i, 'text'] = text

            text_length = len(text.split())
            df_reports.at[i, 'text_length'] = text_length

            # examine language of text
            text_language = detect(text)
            df_reports.at[i, 'text_language'] = text_language

            # check approval
            if total_pages > 10 and text_length > (30 * amount_pages) and text_language == 'en':

                # set approved to yes an leave document in folder
                df_reports.at[i, 'approved'] = 'yes'

                os.replace(document_path, f"data/sustainability_reports_approved/{file_name}")
            else:
                # set approved to no
                df_reports.at[i, 'approved'] = 'no'


            # print to see progress
            print('done: ', i)

            # save df to be able to stop
            if i % 5 == 0:
                df_reports.to_csv('data/reports_filtered_from_1900.csv')
                # set stopping point here

        except Exception as e:
            print(e)
            print('document failed: ', document_path)

    return df_reports


if __name__ == "__main__":

    df_reports = filter_reports_regarding_extractability_and_language(amount_pages)

    # save result
    df_reports.to_csv('data/reports_filtered_from_1900.csv')
