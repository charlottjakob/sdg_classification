# importing required modules
import PyPDF2
import pandas as pd

companies = pd.read_csv('files/un_global_compact_data.csv')
companies['text'] = ""
companies.drop(['cop_link', 'document_file_name', 'file_urls'], axis=1)

for i, company in companies.iterrows():

    try:
        document_path = 'files/cops/' + str(company['document_file_name'])

        # creating a pdf file object
        pdf_file_obj = open(document_path, 'rb')

        # creating a pdf reader object
        pdf = PyPDF2.PdfFileReader(pdf_file_obj)

        # for page_number in range(pdf_reader.numPages):

        #     page_obj = pdf_reader.getPage(page_number)
        #     document_text.append(str(page_obj.extractText()))

        total_page = pdf.numPages
        if total_page > 5:
            total_page = 5
        current_page = 0
        text = ""

        while(current_page < total_page):
            pdf_page = pdf.getPage(current_page)
            try:
                text = text + pdf_page.extractText().replace("\r", "").replace("\n", "")

            except:
                print(current_page)
            current_page += 1
    except:
        print(document_path)

    pdf_file_obj.close()

    companies.at[i, 'text'] = text

companies.to_csv('files/cop_data.csv')