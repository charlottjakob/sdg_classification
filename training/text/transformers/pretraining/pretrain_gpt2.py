# local
from util import PreTrainDataset

# basics
import pandas as pd

# ml
import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# set epochs
EPOCHS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
text = pd.read_csv('data/text_domain.csv')['text'].tolist()

# load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# tokenize
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# create Dataset and DataLoader
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

        # input to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # pass through model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
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
model.save_pretrained('training/text/transformers/models/pretrained_gpt2/')
