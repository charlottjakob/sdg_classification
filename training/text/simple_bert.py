import copy
from torch import nn
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .transformers.utils.helper import class_names, get_default_device
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
from torch import nn, optim
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

MAX_LEN = 500


class BERTClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BERTClassifier, self).__init__()
    self.transformer = BertModel.from_pretrained('bert-base-cased')
    self.drop = nn.Dropout(p=0.1)
    self.linear = nn.Linear(in_features=self.transformer.config.hidden_size, out_features=n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    ).pooler_output

    output = self.drop(output)
    output = self.linear(output)

    return output
  


class TextDataset(Dataset):
  """Dataset for Fine-Tuning."""

  def __init__(self, texts, targets, tokenizer, max_len):
    """Dataset initialization."""
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    """Calcualte length of total dataset."""
    return len(self.texts)

  def __getitem__(self, idx):
    """Get item by its index."""
    # select idx from texts and targets lists
    text = str(self.texts[idx])
    target = self.targets[idx]

    # tokenize
    encoding = self.tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length'
    )

    # return plain text, inputs and targets
    return {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
    }
  

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
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

def eval_model(model, data_loader, loss_fn, device, n_examples):
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
    val_f1 = f1_score(np.array(targets_all), np.array(outputs_all)>0.5, average='micro')

    # return loss, thresholds and f1
    return np.mean(losses), val_f1




def bert_multi_label(X, y):
    """Train model with specific set of parameters."""
    
    # transform parameters for actual application
    epochs = 2
    batch_size = 4 # effective is * 4 =16
    lr = 2e-5
    pos_weight = None
    warmup_ratio = 0.1
    weight_decay = 0.1

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BERTClassifier(len(class_names))

    # get device
    device = get_default_device()

    X_train, X_val, y_train, y_val = train_test_split(X,y.to_numpy(), test_size=0.2, random_state=42)
    X_train.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)


    # print the shapes for validation
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('-' * 20)
    print('X_val shape: ', X_val.shape)
    print('y_val shape: ', y_val.shape)
    print('-' * 20)

    train_ds = TextDataset(texts=X_train, targets=y_train, tokenizer=tokenizer, max_len=MAX_LEN)
    val_ds = TextDataset(texts=X_val, targets=y_val, tokenizer=tokenizer, max_len=MAX_LEN)

    # Create DataLoader according to batch size
    train_data_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=1)
    val_data_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=1)

    # model to GPU
    model = model.to(device)

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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # initialize empty dict for history and best_f1 to be stepped up
    history = defaultdict(list)
    best_f1 = 0

    # for each epoch train and validate the model
    for epoch in range(epochs):

        # Start prints
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # train
        train_loss = train_epoch(
            model=model,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            n_examples=X_train.shape[0]
        )
        print(f'Train Loss: {train_loss}')

        # validate
        val_loss, val_f1 = eval_model(
            model=model,
            data_loader=val_data_loader,
            loss_fn=loss_fn,
            device=device,
            n_examples=X_val.shape[0]
        )
        print(f'Val Loss: {val_loss} F1: {val_f1}')  # np.mean(val_thresholds)
        print()

        # add epochs results to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1.item())


    return history