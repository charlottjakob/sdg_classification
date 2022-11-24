'''process scraped documents for image_classification.

1. cut sdg_pages with sdgs from reports for text_extraction
2. save rest of pages as nosdg_pages

3. calculate how many pages are needed for a balanced dataset
4. add overlays to nosdg_pages
'''

# basics
import random as rn
import os
import pandas as pd

# pdf handling
from fpdf import FPDF
import PyPDF2

# set workingdirectory
# os.chdir('/Users/charlottjakob/Documents/github_repos/sdg_classification')


def create_overlays(o_from, o_to):
    """Creat plain pdf pages including random pacements of random sdgs."""

    # create lost with paths to SDG-Logos
    sdg_strings = [str(sdg) if sdg >= 10 else '0' + str(sdg) for sdg in range(1, 18)]
    sdg_paths = [f'data/SDG_icons_PRINT/E_SDG_PRINT-{sdg_str}.jpg' for sdg_str in sdg_strings]

    # if overlays were already created load dataframe and append
    if os.path.isfile('data/overlays/overlay_labels.csv'):
        page_data = pd.read_csv(('data/overlays/overlay_labels.csv'))
    # if there is not already a file with overlays initialze an empty dataframe
    else:
        page_data = pd.DataFrame()

    # for each number that should be created/updted add random sdgs to emplty page
    for i in range(o_from, o_to):
        print('overlay ', i, ' created')

        # initialize empty page
        pdf = FPDF('L')
        pdf.add_page()

        # initialize weights according to the distribution in the test set e.g. 36 samples include 3 sdg-icons
        # add number for zero SDGs manually to 100 as in sustainability reports most of pages are without SDGs
        weights = [100, 35, 25, 36, 33, 15, 13, 8, 7, 7, 3, 6, 2, 2, 4, 2, 2, 7]

        # get random number between 0 and 18 representing the amount of sdg
        rand_n_sdgs = rn.choices(range(0, 18), weights=weights, cum_weights=None, k=1)[0]
        # get random sdgs
        rand_sdgs = rn.sample(range(0, 17), rand_n_sdgs)

        # if there are more than 6 sdgs on the page take a smaller size-range as the propability of overlapping increases
        if rand_n_sdgs < 6:
            w = rn.randrange(5, 40)
        else:
            w = rn.randrange(5, 20)

        # add sdgs to empty pdf-page
        for rand_sdg in rand_sdgs:

            # get random position
            x = rn.randrange(20, 297 - w)
            y = rn.randrange(20, 210 - w)

            # add sdgs at random position
            pdf.image(sdg_paths[rand_sdg], x, y, w, w)

        # save pdf page includeing SDGs in folder overlays
        pdf.output(f"data/overlays/{i}.pdf", "F")

        # save overlay number and sdgs to dataframe
        page_data.at[i, 'sdgs'] = ','.join([str(rand_sdg + 1) for rand_sdg in rand_sdgs])

    # save dataframe
    page_data.to_csv('data/overlay/overlay_labels.csv')


def add_overlays(f_from=0, f_to=1000, o_from=1000):
    """Add overlays to report pages."""
    # load file_names and overlays
    file_names = pd.read_csv('data/report_pages_nosdg/file_names.csv')['file_name']
    overlay_labels = pd.read_csv('data/overlays/overlay_labels.csv')['sdgs']

    # if train data is already created update it
    if os.path.isfile('data/image_train_labels.csv'):
        train_labels = pd.read_csv(('data/image_train_labels.csv'))
    # else create new DataFrame
    else:
        train_labels = pd.DataFrame()

    # for each report page add overlay
    for i in range(f_from, f_to):

        # get path to file number i
        report_page_path = f'data/report_pages_nosdg/{file_names[i]}.pdf'

        # get overlay starting from number o_from
        sdg_overlay_path = f'data/overlays/{i + o_from}.pdf'
        output_path = f'data/training_pdfs/{i + o_from}.pdf'

        # open documents
        with open(report_page_path, "rb") as underlay, open(sdg_overlay_path, "rb") as overlay:

            # get report page and overlay as document
            background = PyPDF2.PdfFileReader(underlay).getPage(0)
            foreground = PyPDF2.PdfFileReader(overlay).getPage(0)

            # merge overlay to report page
            background.mergePage(foreground)

            # add all pages to a writer
            writer = PyPDF2.PdfFileWriter()
            writer.addPage(background)

            # write everything in the writer to a file
            with open(output_path, "wb") as out_file:
                writer.write(out_file)

        # add sdgs at the documents position in train_labels
        train_labels.at[i + o_from, 'file_name'] = file_names[i]
        train_labels.at[i + o_from, 'sdgs'] = overlay_labels[i + o_from]

        # print for tracking progress
        print('overlay ', i + o_from, ' added to file number ', i)

    # save sdg labels
    train_labels.to_csv('data/image_train_labels.csv')


if __name__ == "__main__":

    create_overlays(o_from=0, o_to=1000)
    add_overlays(f_from=0, f_to=1000, o_from=0)
