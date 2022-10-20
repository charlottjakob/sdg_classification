'''process scraped documents for image_classification.

1. cut sdg_pages with sdgs from reports for text_extraction
2. save rest of pages as nosdg_pages

3. calculate how many pages are needed for a balanced dataset
4. add watermarks to nosdg_pages
'''

from fpdf import FPDF
import PyPDF2
import random as rn
import os
import pandas as pd

os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')


def create_overlays(o_from, o_to):
    """Add SDG logo to plain pdf."""
    sdg_strings = [str(sdg) if sdg >= 10 else '0' + str(sdg) for sdg in range(1, 18)]
    sdg_paths = [f'data/SDG_icons_PRINT/E_SDG_PRINT-{sdg_str}.jpg' for sdg_str in sdg_strings]

    if os.path.isfile('data/watermarks/watermark_labels.csv'):
        page_data = pd.read_csv(('data/watermarks/watermark_labels.csv'))
    else:
        page_data = pd.DataFrame()

    for i in range(o_from, o_to):
        print('overlay ', i, ' created')

        pdf = FPDF('L')
        pdf.add_page()

        weights = [30, 35, 25, 36, 33, 15, 13, 8, 7, 7, 3, 6, 2, 2, 4, 2, 2, 7]
        rand_n_sdgs = rn.choices(range(0, 18), weights=weights, cum_weights=None, k=1)[0]

        rand_sdgs = rn.sample(range(0, 17), rand_n_sdgs)

        # if there are more than 6 sdgs on the page take a smaller size-range
        if rand_n_sdgs < 6:
            w = rn.randrange(5, 40)
        else:
            w = rn.randrange(5, 20)

        for rand_sdg in rand_sdgs:

            x = rn.randrange(20, 297 - w)
            y = rn.randrange(20, 210 - w)
            pdf.image(sdg_paths[rand_sdg], x, y, w, w)

        pdf.output(f"data/watermarks/{i}.pdf", "F")
        page_data.at[i, 'sdgs'] = ','.join([str(rand_sdg + 1) for rand_sdg in rand_sdgs])

    page_data.to_csv('data/watermarks/watermark_labels.csv')


def add_overlays(f_from=0, f_to=1000, o_from=1000):
    """2. add pdf with logo on top of report_page."""
    file_names = pd.read_csv('data/report_pages_nosdg/file_names.csv')['file_name'] # 1000
    watermark_labels = pd.read_csv('data/watermarks/watermark_labels.csv')['sdgs']  # 3000

    if os.path.isfile('data/image_train_labels.csv'):
        train_labels = pd.read_csv(('data/image_train_labels.csv'))
    else:
        train_labels = pd.DataFrame()

    for i in range(f_from, f_to):

        report_page_path = f'data/report_pages_nosdg/{file_names[i]}.pdf'  # takes 0 - 1000
        sdg_overlay_path = f'data/watermarks/{i + o_from}.pdf'  # take 1000 - 2000
        output_path = f'data/training_pdfs/{i + o_from}.pdf'  # outputs 1000 - 2000

        with open(report_page_path, "rb") as underlay, open(sdg_overlay_path, "rb") as overlay:
            original = PyPDF2.PdfFileReader(underlay)
            background = original.getPage(0)
            foreground = PyPDF2.PdfFileReader(overlay).getPage(0)

            # merge the first two pages
            background.mergePage(foreground)

            # add all pages to a writer
            writer = PyPDF2.PdfFileWriter()
            for j in range(original.getNumPages()):
                page = original.getPage(j)
                writer.addPage(page)

            # write everything in the writer to a file
            with open(output_path, "wb") as out_file:
                writer.write(out_file)

        train_labels.at[i + o_from, 'file_name'] = file_names[i]
        train_labels.at[i + o_from, 'sdgs'] = watermark_labels[i + o_from]
        print('overlay ', i + o_from, ' added to file number ', i)

    train_labels.to_csv('data/image_train_labels.csv')

# create_overlays(o_from=1000, o_to=2000)
# add_overlays(f_from=0, f_to=1000, o_from=1000)  # files=(0-1000),overlays=(1000, 2000)
