# local
from util import PreTrainDataset

import pandas as pd
import random

# ml

import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForPreTraining

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 1  # 10

# load data
text = pd.read_csv('data/text_domain.csv')['text'][:10]

# download model and tokenizer
model = BertForPreTraining.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# create list of sentences
bag = [item for sentence in text for item in sentence.split('.') if item != '']
bag_size = len(bag)

sentence_a = []
sentence_b = []
label = []

# for each report page create a sentence pair for Next Sentence Prediction
for paragraph in text:
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences - 2)

        # 50% are sentences that follow one another
        if random.random() >= 0.5:
            sentence_a.append(sentences[start])  # random sentence form that page
            sentence_b.append(sentences[start + 1])  # the following sentence
            label.append(0)

        # 50% are random sentences
        else:
            index = random.randint(0, bag_size - 1)
            sentence_a.append(sentences[start])  # random sentence form that page
            sentence_b.append(bag[index])  # add random sentence for somewhere in the dataset
            label.append(1)

# tokenize
inputs = tokenizer(
    sentence_a,
    sentence_b,
    return_tensors='pt',
    max_length=512,
    truncation=True,
    padding='max_length'
)

# set labels for next sentence prediction
# 1: Sentence B follows sentence A, 0: not
inputs['next_sentence_label'] = torch.LongTensor([label]).T

# Masked Language Modeling
# Copy Text from Next Sentence prediction
inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)

# set berts specific tokens
token_numbers = {'cls': 101, 'sep': 102, 'pad': 0, 'mask': 103}

# create mask array
# take care to not mask CLS (101), SEP (102), and PAD (0)
mask_arr = (rand < 0.15) * (inputs.input_ids != token_numbers['cls']) * (inputs.input_ids != token_numbers['sep']) * (inputs.input_ids != token_numbers['pad'])

selection = []
# get indices of Trues (indices of to be masked words)
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
# set value 103 instead of input_ids at inices
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = token_numbers['mask']


# creat Dataset and Dataloader
dataset = PreTrainDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

# model to GPU and to train mode
model.to(device)
model.train()

# set optimizer
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(EPOCHS):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:

        # initialize calculated gradients (from prev step)
        optim.zero_grad()

        # inputs to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)

        # pass through model
        outputs = model(
            input_ids,  # input_ids with masked: 103
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # indication 0: sentence A, 1: sentence B
            next_sentence_label=next_sentence_label,  # indication 1: stentence B follows sentence A
            labels=labels  # all input_ids without masked words
        )

        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# save model after training
model.save_pretrained('training/text/transformers/models/pretrained_bert/')
