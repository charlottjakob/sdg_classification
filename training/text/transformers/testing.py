# local functions
from utils.dataset import TextDataset
from utils.train_evaluate import test
from utils.helper import get_default_device
from utils.model import BERTClassifier, XLNetClassifier, GPT2Classifier # needed for loading models
from torch.utils.data import DataLoader

# basics
import pandas as pd
import numpy as np
import json

# ml
import torch
from transformers import RobertaTokenizer, BertTokenizer, GPT2Tokenizer, XLNetTokenizer


# choose models and training data
MODEL_NAMES = ['bert', 'gpt2', 'roberta', 'xlnet']
DATA_NUMBERS = [2]

if __name__ == '__main__':

    for model_name in MODEL_NAMES:

        # reset DataFrame for predictions
        df_preds = pd.DataFrame()
        df_preds.to_csv(f'data/text_test_results/{model_name}.csv')

        # get device
        device = get_default_device()

        # load testing data
        df_test = pd.read_csv('data/text_test.csv')

        # download tokenizer according to transformer model
        if model_name == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        elif model_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        elif model_name == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        # Build X and y
        X_test = df_test['text']
        class_names = [str(number) for number in np.arange(1, 18)]
        y_test = df_test[class_names].to_numpy()

        # create dataloader
        test_ds = TextDataset(texts=X_test, targets=y_test, tokenizer=tokenizer, max_len=512)
        test_data_loader = DataLoader(test_ds, batch_size=5, num_workers=2)

        # Create DataFrame with targets
        df_preds = pd.read_csv(f'data/text_test_results/{model_name}.csv', index_col=0)
        df_preds['target'] = y_test.tolist()

        # for both datasets predict sdgs and add to DataFrame
        for data_number in DATA_NUMBERS:

            # get history
            with open(f'data/histories/{data_number}_{model_name}.json') as f:
                history = json.load(f)[f'{data_number}_{model_name}']

            # get threshold from epoch where model was saved
            epoch_saved = np.argmax(history['final']['val_f1'])
            thresholds = history['final']['threshold'][epoch_saved]

            # load model
            model = torch.load(f'training/text/transformers/models/{data_number}_{model_name}.model', map_location=device)  # torch.device('cpu')

            # predict testing data
            targets, outputs, predictions = test(
                model,
                test_data_loader,
                device,
                thresholds
            )

            # add predictions to DataFrame
            df_preds[f'pred_{data_number}'] = predictions.tolist()
            df_preds[f'outs_{data_number}'] = outputs.tolist()

        # Save predictions in DataFrame
        df_preds.to_csv(f'data/text_test_results/{model_name}.csv')
