"""Save labeled pdf-pages as single pdf files in folder single_pages."""

# basic
import pandas as pd
import os
import random

# page extraction
import PyPDF2


# load table with number of pages and according labels
os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')


def seperate_report_pages(n_nosdg_pages):
    label_data = pd.read_csv('data/report_labeling.csv')

    # delete pdfs that are not reports
    label_data = label_data[label_data['comment'] != 'no report']
    label_data = label_data[label_data['comment'] != 'can not be opened']

    file_names = label_data['file_name'].unique()

    n_nosdg_pages_per_report = int(n_nosdg_pages / (len(file_names) * 0.7)) + 1

    # create DataFrame for saving name of nosdg pages
    nosdg_page_df = pd.DataFrame(columns=['file_name'])
    sdg_page_df = pd.DataFrame(columns=['file_name', 'sdgs'])

    for file_idx, file_name in enumerate(file_names):
        sdg_pages = label_data[(label_data['file_name'] == file_name) & (label_data['page'].notna())]
        sdg_page_numbers = list(sdg_pages['page'].astype('Int64'))

        document_path = f'data/sustainability_reports/{file_name}'

        if os.path.isfile(document_path):
            # creating a pdf file object
            pdf_file_obj = open(document_path, 'rb')

            try:
                # creating a pdf reader object
                pdf = PyPDF2.PdfFileReader(pdf_file_obj)
                amount_pages = pdf.numPages

                # Get page_numbers for nosdg pages (exclude all labeled pages)
                if amount_pages > n_nosdg_pages_per_report + len(sdg_page_numbers):
                    n = n_nosdg_pages_per_report
                else:
                    # if not enough pages in report take as much as possible
                    n = amount_pages - len(sdg_page_numbers)
                nosdg_page_numbers = [number for number in random.sample(range(2, amount_pages), n + len(sdg_page_numbers)) if number not in sdg_page_numbers][:n]

                # Get page_numbers for sdg pages -> exclude reversed sdgs
                sdg_colored_page_numbers = sdg_pages[sdg_pages['sdgs'].notna()]

                # extract pages with sdgs and save in report_pages_sdg
                for idx, row in sdg_colored_page_numbers.iterrows():

                    page_number = int(row['page'])
                    page = pdf.getPage(page_number - 1)

                    output = PyPDF2.PdfFileWriter()
                    output.addPage(page)

                    page_path = f'data/report_pages_sdg/{file_name}_{page_number}.pdf'
                    with open(page_path, "wb") as output_stream:
                        output.write(output_stream)

                    sdg_page_df = sdg_page_df.append({'file_name': str(file_name) + '_' + str(page_number), 'sdgs': row['sdgs']}, ignore_index=True)

                # # extract pages without sdgs and save in report_pages_nosdg
                # for page_number in nosdg_page_numbers:

                #     page = pdf.getPage(page_number - 1)

                #     output = PyPDF2.PdfFileWriter()
                #     output.addPage(page)

                #     page_path = f'data/report_pages_nosdg/{file_name}_{page_number}.pdf'
                #     with open(page_path, "wb") as output_stream:
                #         output.write(output_stream)

                #     nosdg_page_df = nosdg_page_df.append({'file_name': str(file_name) + '_' + str(page_number)}, ignore_index=True)

            except Exception as e:
                print(e)
                print('failed: ', file_name)

        else:
            print('report not in documents: ' + str(file_name))

        print((file_idx + 1), '/', len(file_names))

    # nosdg_page_df.to_csv('data/report_pages_nosdg/file_names.csv')
    sdg_page_df.to_csv('data/image_test_labels.csv')

