
# locals
from .dataset import TextDataset
from .model import BERTClassifier, XLNetClassifier, GPT2Classifier
from .helper import get_f1_with_optimal_thresholds, class_names, MAX_LEN

# basics
from collections import defaultdict
import numpy as np
import copy

# ml
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import XLNetTokenizer, XLNetModel


class TrainingClass():
    """"""
    def __init__(self, device, X_train, y_train, X_val, y_val, MODEL_NAME='bert', further_pretrained=False, NUMBER_TRAIN_DATA=1, label_weights=None):

        # save varibales for training
        self.MAX_LEN = MAX_LEN
        self.len_train = len(X_train)
        self.len_val = len(X_val)
        self.device = device
        self.NUMBER_TRAIN_DATA = NUMBER_TRAIN_DATA
        self.MODEL_NAME = MODEL_NAME
        self.label_weights = label_weights

        # load tokenizer and model
        if MODEL_NAME == 'roberta':

            # load tokenizer from remote
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

            # Load Pre-trained from remote or Further-Pre-trained from local
            if further_pretrained is True:
                self.transformer_layers = RobertaModel.from_pretrained(f'training/text/transformers/models/pretrained_{NUMBER_TRAIN_DATA}_{MODEL_NAME}/')
            else:
                self.transformer_layers = RobertaModel.from_pretrained("roberta-base")

        if MODEL_NAME == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            if further_pretrained is True:
                self.transformer_layers = BertModel.from_pretrained(f'training/text/transformers/models/pretrained_{NUMBER_TRAIN_DATA}_{MODEL_NAME}')
            else:
                self.transformer_layers = BertModel.from_pretrained('bert-base-cased')

        if MODEL_NAME == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            if further_pretrained is True:
                self.transformer_layers = GPT2Model.from_pretrained(f'drive/MyDrive/Master Thesis/models/pretrained_{NUMBER_TRAIN_DATA}_{MODEL_NAME}')
            else:
                self.transformer_layers = GPT2Model.from_pretrained("gpt2")

        if MODEL_NAME == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            if further_pretrained is True:
                self.transformer_layers = XLNetModel.from_pretrained(f'drive/MyDrive/Master Thesis/models/pretrained_{NUMBER_TRAIN_DATA}_{MODEL_NAME}')
            else:
                self.transformer_layers = XLNetModel.from_pretrained('xlnet-base-cased')

        # Create Datasets
        self.train_ds = TextDataset(texts=X_train, targets=y_train, tokenizer=tokenizer, max_len=MAX_LEN)
        self.val_ds = TextDataset(texts=X_val, targets=y_val, tokenizer=tokenizer, max_len=MAX_LEN)

    def train(self, parameters, save_model=False, trial=None):
        """Train model with specific set of parameters."""
        # transform parameters for actual application
        epochs = parameters['epochs']
        batch_size = int(parameters['batch_size'] / 4)  # transform effective batch size to actual batch size
        lr = parameters['lr']
        pos_weight = self.label_weights if parameters['with_weight'] is True else None  # set weigths for cost sensitive learning if with_weight is True
        warmup_ratio = parameters['warmup_ratio']
        weight_decay = parameters['weight_decay']

        # Create DataLoader according to batch size
        train_data_loader = DataLoader(self.train_ds, batch_size=batch_size, num_workers=2)
        val_data_loader = DataLoader(self.val_ds, batch_size=batch_size, num_workers=2)

        # initialize model with previously initialized Transformer Layers
        if self.MODEL_NAME == 'roberta' or self.MODEL_NAME == 'bert':
            model = BERTClassifier(self.transformer_layers, len(class_names))
        elif self.MODEL_NAME == 'xlnet':
            model = XLNetClassifier(self.transformer_layers, len(class_names))
        elif self.MODEL_NAME == 'gpt2':
            model = GPT2Classifier(self.transformer_layers, len(class_names))

        # model to GPU
        model = model.to(self.device)

        # set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # set scheduler
        total_steps = int(len(train_data_loader) / 4 * epochs)  # because gradient accumulation -> scheduler is called after every 4  actual batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )

        # set loss function
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)

        # initialize empty dict for history and best_f1 to be stepped up
        history = defaultdict(list)
        best_f1 = 0

        # for each epoch train and validate the model
        for epoch in range(epochs):

            # Start prints
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            # train
            train_loss = self.train_epoch(
                model=model,
                data_loader=train_data_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=self.device,
                scheduler=scheduler,
                n_examples=self.len_train
            )
            print(f'Train Loss: {train_loss}')

            # validate
            val_loss, val_thresholds, val_f1 = self.eval_model(
                model=model,
                data_loader=val_data_loader,
                loss_fn=loss_fn,
                device=self.device,
                n_examples=self.len_val
            )
            print(f'Val Loss: {val_loss} Thresholds: {val_thresholds} F1: {val_f1}')  # np.mean(val_thresholds)
            print()

            # add epochs results to history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1.item())
            history['threshold'].append(val_thresholds)

            # save epoch if F1 improved
            if val_f1 > best_f1:
                best_f1 = val_f1
                if save_model is True:
                    torch.save(model, f'drive/MyDrive/Master Thesis/models/{self.NUMBER_TRAIN_DATA}_{self.MODEL_NAME}.model')

        # after training finish history by adding parameters
        history['parameters'] = copy.deepcopy(parameters)

        # delete dataloaders
        del train_data_loader
        del val_data_loader

        return history

    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        """Train a single epoch."""
        # set model to train mode
        model = model.train()

        # initialize loss list and the degree of Gradient Accumulation
        losses = []
        accum_iter = 4

        # for each batch do forward pass for every 4th batch do backword pass
        for batch_idx, d in enumerate(data_loader):

            # input to GPU
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # turn on gradient calculation
            with torch.set_grad_enabled(True):

                # forward pass without sigmoid
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # calculate loss
                loss_batch = loss_fn(outputs, targets.float())
                losses.append(loss_batch.item())

                # calculate loss proportion for next backword pass
                loss_prop = loss_batch / accum_iter

                # add the batchs gradients to accumulated gradients
                loss_prop.backward()

                # for every 4th batch or the last batch run optimization
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):

                    # update Parameters with accumulated gradients and optimize
                    optimizer.step()

                    # report to scheduler that step is done so it updates the lr
                    scheduler.step()

                    # set accumulated gradients to zero
                    optimizer.zero_grad()

        # return losses
        return np.mean(losses)

    def eval_model(self, model, data_loader, loss_fn, device, n_examples):
        """Evaluate a single epoch."""
        
        # ste model to evaluation mode
        model = model.eval()

        # initialization
        losses = []
        outputs_all = []
        targets_all = []
        correct_predictions = 0

        # stop keeping track of gradients
        with torch.no_grad():

            # for each batch get scores
            for d in data_loader:

                # inputs to GPU
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                # forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # bring outputs to the scale of 0,1 (because in loss function sigmoid is alread included)
                outputs_sm = torch.sigmoid(outputs)

                # save ouptuts and targets to calculate F1 after all batches done
                outputs_all.extend(outputs_sm.cpu().tolist())
                targets_all.extend([target.tolist() for target in targets.cpu()])

                # loss
                loss = loss_fn(outputs, targets.float())
                losses.append(loss.item())

        # get best thresholds and F1-scores
        val_f1, val_thresholds = get_f1_with_optimal_thresholds(np.array(outputs_all), np.array(targets_all))

        # return loss, thresholds and f1
        return np.mean(losses), val_thresholds, val_f1  # correct_predictions.double() /(n_examples*17)


def test(model, data_loader, device, thresholds):
    """Predict text with given model, data and thresholds."""
    # set model to evaluation mode
    model = model.eval()

    # initialize empty lists to be filled
    preds = []
    targets = []
    outs = []

    # stop keeping track of gradients
    with torch.no_grad():

        # for each batch get ouputs and calculate predictions
        for d in data_loader:

            # inputs to GPU
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets_batch = d["targets"].to(device)

            # forward pass
            outs_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # bring output to range of 0-1
            outs_batch = torch.sigmoid(outs_batch)

            # divide ouptuts according to the labels thresholds
            preds_batch = (outs_batch > torch.tensor(thresholds)).float()

            # save targets, outputs and preds
            targets.extend(targets_batch.tolist())
            outs.extend(outs_batch.tolist())
            preds.extend(preds_batch.tolist())

    # retur targets, outputs and preds
    return np.array(targets), np.array(outs), np.array(preds)
