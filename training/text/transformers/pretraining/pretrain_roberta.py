# local
from utils.dataset import PreTrainDataset

# basics
import pandas as pd

# ml
import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import RobertaTokenizer, RobertaForMaskedLM


# set epochs
EPOCHS = 10

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
text = pd.read_csv('data/text_domain.csv')['text'][:10]

# load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# tokenize text
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# create labels as copy from input_ids as it labels are all unmasked words
inputs['labels'] = inputs.input_ids.detach().clone()

# Mask 15% of the input ids
token_numbers = {'cls': 0, 'sep': 2, 'pad': 1, 'mask': 50264}

# radomly create numbers between 0 and 1
rand = torch.rand(inputs['input_ids'].shape)

# by selecte all numbers samller than 0.15 we get 15%
# furthermore we exclude special tokens from maskting
mask_arr = (rand < 0.15) * (inputs['input_ids'] != token_numbers['cls']) * (inputs['input_ids'] != token_numbers['sep']) * (inputs['input_ids'] != token_numbers['pad'])

selection = []
# get indices of Trues (indices of to be masked words)
for i in range(inputs['input_ids'].shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
# set mask token instead of input_ids at inices
for i in range(inputs['input_ids'].shape[0]):
    inputs['input_ids'][i, selection[i]] = token_numbers['mask']


# create Dataset and DataLoader
dataset = PreTrainDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

# model to GPU and to train
model.to(device)
model.train()

# set optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# for each epoch train all batches
for epoch in range(EPOCHS):

    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)

    for batch in loop:

        # initialize calculated gradients
        optim.zero_grad()

        # inputs to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # pass through model
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # extract loss
        loss = outputs.loss

        # add loss where gradient get updated
        loss.backward()

        # update parameters
        optim.step()

        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# after training save model
model.save_pretrained(f'training/text/transformers/models/pretrained_roberta/')