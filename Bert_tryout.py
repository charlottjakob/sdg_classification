import transformers
import optuna
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import regex as re

import nltk
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 160
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def text_preprocessing(df):
  # only keep letters and spaces
  df['text'] = df['text'].apply(lambda x: re.sub('[^A-Za-z\s\']+', ' ', x))
  df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

  # Lower Case
  df['text'] = df['text'].apply(lambda x: x.lower())

  # Stop Words
  df['text'] = df['text'].apply(lambda x: " ".join([word for word in x.split(' ') if word not in stopwords]) )

  return df





class GPReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.reviews)
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation=True,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.text.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    # num_workers=1
  )


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(in_features=self.bert.config.hidden_size, out_features=n_classes)
    self.softmax = nn.Softmax(dim=1)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)  
    output = self.out(output)
    return self.softmax(output)

def train_epoch(
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
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
    )  
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
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
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)


def objective(trial):


  # Test 
  df_test = pd.read_csv('data/Test_data_course.csv')
  df_test = df_test[['text', 'SDG']]
  df_test = df_test.rename(columns={'SDG': 'sdg'})
  df_test['label'] = df_test['sdg'].astype(int) -1
  df_test = text_preprocessing(df_test)
  # df_test['label'] = df_test['sdg'].apply(lambda x: label_to_list(x))

  # Train
  df_train = pd.read_csv('data/Train_data_course.csv')
  df_train = df_train[['text', 'SDG']]
  df_train = df_train.rename(columns={'SDG': 'sdg'})
  df_train['label'] = df_train['sdg'].astype(int) -1
  df_train = text_preprocessing(df_train)
  # df_train['label'] = df_train['sdg'].apply(lambda x: label_to_list(x))

  class_names = [str(number) for number in np.arange(1,18)]

  # df_val, df_test = train_test_split(
  #   df_test,
  #   test_size=0.5,
  #   random_state=RANDOM_SEED
  # )

  print('df_train shape: ', df_train.shape)
  print('df_train head: ' )
  print(df_train.head())
  print('-'*20)
  print('df_test shape: ', df_test.shape)
  print('df_test head: ' )
  print(df_test.head())




  # 'batch_size': 40, 'lr': 0.2519355724425978 -> 0.711 (5 epochs)
  BATCH_SIZE = trial.suggest_int("batch_size", 10, 40, step=10)
  lr = trial.suggest_float("lr", 0.01, 1, log=True)
  
  train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
  #val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
  test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

  model = SentimentClassifier(len(class_names))
  model = model.to(device)

  EPOCHS = 5
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
   )
  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0
  for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
      model,
      test_data_loader,
      loss_fn,
      device,
      len(df_test)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc
  return val_acc


if __name__ == '__main__':

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=5)

  trial = study.best_trial

  print('Accuracy: {}'.format(trial.value))
  print("Best hyperparameters: {}".format(trial.params))