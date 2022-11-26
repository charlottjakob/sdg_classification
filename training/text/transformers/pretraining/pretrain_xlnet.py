# local
from util import PreTrainDataset

# basics
import pandas as pd
import random

# ml
import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import XLNetTokenizer, XLNetLMHeadModel

# set epochs
EPOCHS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
text = pd.read_csv('data/text_domain.csv')['text'].tolist()

# download model and tokenizer
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# tokenize
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# define amount of tokens that shlould be masked per data point
mask_tokens_per_sequence = 20

# save number ob samples and text length
n_samples = inputs.input_ids.shape[0]
sequence_length = inputs.input_ids.shape[1]

# randomly select same amount of tokens per sequence
masks = torch.tensor([sorted(random.sample(range(1, 510), mask_tokens_per_sequence)) for x in range(n_samples)])

# set masks to 1 in perm_mask that all other words cannot see these tokens
inputs['perm_mask'] = torch.zeros((n_samples, sequence_length, sequence_length), dtype=torch.float)
for i in range(n_samples):
    inputs['perm_mask'][i, :, masks[i]] = 1.0

# add masks in accending order to target_mapping that they will be predicted one by one
inputs['target_mapping'] = torch.zeros((n_samples, mask_tokens_per_sequence, sequence_length), dtype=torch.float)
for i, i_mask in enumerate(masks):
    for j in range(mask_tokens_per_sequence):
        inputs['target_mapping'][i, j, i_mask[j]] = 1.0

# create labels by looping through masks and selecting the true input_ids
labels = []
for i in range(n_samples):
    labels.append(inputs.input_ids[i, masks[i]].tolist())
inputs['labels'] = torch.tensor(labels)


# print shapes of inputs to verify correctness
print('input_ids: ', inputs['input_ids'].shape)
print('perm_mask: ', inputs['perm_mask'].shape)
print('target_mapping: ', inputs['target_mapping'].shape)
print('label: ', inputs['labels'].shape)

# Create Dataset and DataLoader
dataset = PreTrainDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

# model to GPU and to train mode
model.to(device)
model.train()

# set optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# loop through epochs and batchs
for epoch in range(EPOCHS):

    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)

    # loop through batches
    for batch in loop:

        # initialize calculated gradients (from prev step)
        optim.zero_grad()

        # inputs to GPU
        input_ids = batch['input_ids'].to(device)
        perm_mask = batch['perm_mask'].to(device)
        target_mapping = batch['target_mapping'].to(device)
        labels = batch['labels'].to(device)

        # pass through model
        outputs = model(
            input_ids=input_ids,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            labels=labels
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
model.save_pretrained('training/text/transformers/models/pretrained_xlnet/')
