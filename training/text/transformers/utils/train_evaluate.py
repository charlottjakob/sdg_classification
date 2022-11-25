from .dataset import TextDataset
from .model import BERTClassifier, XLNetClassifier, GPT2Classifier
from .helper import get_f1_with_optimal_thresholds
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import optuna
import copy
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import XLNetTokenizer, XLNetModel
class_names = [str(number) for number in np.arange(1, 18)]


class TrainingClass():
    def __init__(self, device, X_train, y_train, X_val, y_val, MAX_LEN, MODEL_NAME='bert', further_pretrained=False, NUMBER_TRAIN_DATA=1, label_weights=None):

        self.MAX_LEN = MAX_LEN
        self.len_train = len(X_train)
        self.len_val = len(X_val)
        self.device = device
        self.NUMBER_TRAIN_DATA = NUMBER_TRAIN_DATA
        self.MODEL_NAME = MODEL_NAME
        self.label_weights = label_weights

        if MODEL_NAME == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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

        self.train_ds = TextDataset(texts=X_train, targets=y_train, tokenizer=tokenizer, max_len=MAX_LEN)
        self.val_ds = TextDataset(texts=X_val, targets=y_val, tokenizer=tokenizer, max_len=MAX_LEN)

    def train(self, parameters, save_model=False, trial=None):

        # transform parameters for actual application
        epochs = parameters['epochs']
        batch_size = int(parameters['batch_size'] / 4)  # transform effective batch size to actual batch size
        lr = parameters['lr']
        pos_weight = self.label_weights if parameters['with_weight'] is True else None
        warmup_ratio = parameters['warmup_ratio']
        weight_decay = parameters['weight_decay']

        train_data_loader = DataLoader(self.train_ds, batch_size=batch_size, num_workers=2)
        val_data_loader = DataLoader(self.val_ds, batch_size=batch_size, num_workers=2)

        # model = SDGClassifier(dropout_rate, len(class_names), is_gpt=is_gpt)
        if self.MODEL_NAME == 'roberta' or self.MODEL_NAME == 'bert':
            model = BERTClassifier(self.transformer_layers, len(class_names))
        elif self.MODEL_NAME == 'xlnet':
            model = XLNetClassifier(self.transformer_layers, len(class_names))
        elif self.MODEL_NAME == 'gpt2':
            model = GPT2Classifier(self.transformer_layers, len(class_names))

        model = model.to(self.device)

        # optimizer = AdamW(model.parameters(), lr=lr,  weight_decay=weight_decay, correct_bias=False)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = int(len(train_data_loader) / 4 * epochs)  # because gradient accumulation
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )
        # loss_fn = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)

        history = defaultdict(list)
        best_f1 = 0
        best_epoch = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch(
                model=model,
                data_loader=train_data_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=self.device,
                scheduler=scheduler,
                n_examples=self.len_train
            )
            print(f'Train Loss: {train_loss}')
            val_loss, val_thresholds, val_f1 = self.eval_model(
                model=model,
                data_loader=val_data_loader,
                loss_fn=loss_fn,
                device=self.device,
                n_examples=self.len_val
            )
            print(f'Val Loss: {val_loss} Thresholds: {val_thresholds} F1: {val_f1}')  # np.mean(val_thresholds)
            print()

            # history['train_acc'].append(train_acc.item())
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1.item())
            history['threshold'].append(val_thresholds)

            if val_f1 > best_f1:
                best_epoch = epoch
                best_f1 = val_f1
                if save_model is True:
                    torch.save(model, f'drive/MyDrive/Master Thesis/models/{self.NUMBER_TRAIN_DATA}_{self.MODEL_NAME}.model')

        history['parameters'] = copy.deepcopy(parameters)
        history['epoch_saved'] = best_epoch

        del train_data_loader
        del val_data_loader

        return history

    def train_epoch(
        self,
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
    ):
        model = model.train()
        losses = []
        outputs_all = []
        correct_predictions = 0
        accum_iter = 4
        for batch_idx, d in enumerate(data_loader):

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            with torch.set_grad_enabled(True):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                outputs_sm = torch.sigmoid(outputs)
                preds = (outputs_sm > 0.5).float()  # _, preds = torch.max(outputs, dim=1)
                outputs_all.extend(outputs_sm.cpu())
                correct_predictions += torch.sum(preds == targets)

                loss_batch = loss_fn(outputs, targets.float())
                losses.append(loss_batch.item())

                loss_prop = loss_batch / accum_iter
                loss_prop.backward()

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        return correct_predictions.double() / (n_examples * 17), np.mean(losses)

    def eval_model(self, model, data_loader, loss_fn, device, n_examples):
        model = model.eval()
        losses = []
        outputs_all = []
        targets_all = []
        correct_predictions = 0
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # _, preds = torch.max(outputs, dim=1)
                outputs_sm = torch.sigmoid(outputs)
                # f1
                outputs_all.extend(outputs_sm.cpu().tolist())
                targets_all.extend([target.tolist() for target in targets.cpu()])

                # acc
                preds = (outputs_sm > 0.5).float()
                correct_predictions += torch.sum(preds == targets)

                # loss
                loss = loss_fn(outputs, targets.float())
                losses.append(loss.item())

        # val_f1 = F_score(outputs_all, targets_all)
        val_f1, val_tresholds = get_f1_with_optimal_thresholds(np.array(outputs_all), np.array(targets_all))

        return np.mean(losses), val_tresholds, val_f1  # correct_predictions.double() /(n_examples*17)


def test(model, data_loader, device, thresholds):
    model = model.eval()
    preds = []
    targets = []
    outs = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets_batch = d["targets"].to(device)
            outs_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            outs_batch = torch.sigmoid(outs_batch)
            preds_batch = (outs_batch > torch.tensor(thresholds)).float()

            targets.extend(targets_batch.tolist())
            outs.extend(outs_batch.tolist())
            preds.extend(preds_batch.tolist())

    return np.array(targets), np.array(outs), np.array(preds)
