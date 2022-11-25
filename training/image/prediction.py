import torch
from utils.train_evaluate import predict
from utils.helper import get_default_device, to_device
from utils.dataset import PredictionDataset, DeviceDataLoader
from torch.utils.data import DataLoader
from utils.model import ImageClassifier

import os
import PyPDF2
import pandas as pd
import numpy as np


import torch.nn as nn
class ResNet15(nn.Module):
    pass


MODEL_NAME = 'image'

# load data
file_names = pd.read_csv('data/reports_approval.csv')

# check if GPU is available
device = get_default_device()
print('device: ', device)

# load model
model = torch.load(f'training/image/{MODEL_NAME}.model', map_location=device)
model = to_device(model, device)


# filter for file_names that were approved (readable, in english)
file_names = file_names[file_names['approved'] == 'yes']

# make sure that file_names are unique
file_names = file_names['document_file_name'].unique()[:2]

# create empty dataframe to later fill it with positive predictions
df = pd.DataFrame(columns=['file_name', 'page', 'predictions'])

# iterate through sustainability reports and predict them
for i, file_name in enumerate(file_names):

    # create path to document
    file_path = 'data/sustainability_reports_500_1500/' + str(file_name)

    # if document is available predict it's pages
    if os.path.isfile(file_path):

        try:
            # prepare annotations list
            pdf = PyPDF2.PdfFileReader(file_path)
            amount_pages = pdf.getNumPages()

            # prepare dataset
            pred_set = PredictionDataset(file_name=file_name, transform=True)
            pred_loader = DataLoader(pred_set, batch_size=1)
            pred_dl = DeviceDataLoader(pred_loader, device)

            # predict single pages
            outputs = predict(model, pred_dl)

            predictions_strs = [','.join([str(x + 1) for x in np.where(np.array(output) > 0.5)[0]]) for output in outputs]
            df = df.append(pd.DataFrame({
                'file_name': [file_name] * amount_pages,
                'page': range(amount_pages),
                'predictions': predictions_strs}), ignore_index=True)

            del pred_set
            print('done: ', i)

        except Exception as e:
            print(e)
            print('failed: ', file_name)

    else:
        print('Path not found: ', file_name)

df.to_csv('data/reports_prediction.csv')
