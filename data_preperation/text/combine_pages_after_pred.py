"""Extract full text from pages with predicted SDGs."""

# basic
import pandas as pd
import os
import numpy as np

# text extraction
from PyPDF2 import PdfFileWriter, PdfFileReader
os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')

# [0,]

# csv input name
pdfs_from = 750

# csv output add on
start_at = 170


label_data = pd.read_csv(f'data/prediction_results_{pdfs_from}.csv')

# filter for pages with sdgs
pages_sdgs = label_data[label_data['predictions'].notna()]
pages_sdgs.reset_index(drop=True, inplace=True)

# get file_names
file_names = label_data['file_name'].unique()[start_at:]

# index = np.argwhere((file_names == '1596740398'))
# file_names = np.delete(file_names, index)

# initialize output documents
output_pdf = PdfFileWriter()
# j = 0
page_count = 0
# pdf_count = 0
output_csv = pd.DataFrame()

for idx, file_name in enumerate(file_names):

  file_number = start_at + idx

  try:
    pages = pages_sdgs[pages_sdgs['file_name'] == file_name]
    document_path = 'data/sustainability_reports_approved/' + str(file_name)

    pdf_file_obj = open(document_path, 'rb')
    pdf = PdfFileReader(pdf_file_obj, strict=False)
    amount_pages = pdf.getNumPages()
    print('file_number: ', file_number, 'amount pages: ', pdf.getNumPages(), 'file_name: ', file_name)

    if not pages.empty and not pdf.isEncrypted and amount_pages < 250:
      for _, page in pages.iterrows():
        page_number = page['page']
        pdf_page = pdf.getPage(page_number)
        output_pdf.addPage(pdf_page)

        output_csv = output_csv.append({'file_name': str(int(file_name)), 'page': str(int(page_number)), 'predictions': page['predictions']},ignore_index=True)

        page_count += 1

    # # save pdf if more than 900 pages
    if page_count >= 800 or idx >= len(file_names) - 1:
      break

  except Exception as e:
    print(e)
    print('failed: ', document_path)



# save pdf
with open(f"data/pages_sdgs_verified/sdgs_predicted_pdf_{pdfs_from + start_at}.pdf", "wb") as output_stream:
    output_pdf.write(output_stream)

# save csv
output_csv['page_in_compliation'] = range(1, len(output_csv) + 1)
output_csv.to_csv(f'data/pages_sdgs_verified/sdgs_predicted_list_{pdfs_from + start_at}.csv')


if idx >= len(file_names) - 1:
  print('done')
else:
  print('start with file_name number: ', file_number + 1)

    #   # save pdf and create new instance
    #   with open(f"data/pages_sdgs_verified/sdgs_predicted_pdf_{pdfs_from + idx}.pdf", "wb") as output_stream:
    #       output_pdf.write(output_stream)
    #   output_pdf = PdfFileWriter()

    #   # save csv and create new instance
    #   output_csv['page_in_compliation'] = range(1, len(output_csv) + 1)
    #   output_csv.to_csv(f'data/pages_sdgs_verified/sdgs_predicted_list_{pdfs_from + idx}.csv')
    #   output_csv = pd.DataFrame()

    #   page_count = 0
    #   pdf_count += 1