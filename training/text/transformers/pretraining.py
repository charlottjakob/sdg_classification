import pandas as pd
import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForPreTraining
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text = pd.read_csv('data/text_domain.csv')[:10]
text['text'] = text['text'].apply(lambda x: re.sub('[^A-Za-z\s\.]+', ' ', x))  # \'
text['text'] = text['text'].apply(lambda x: re.sub('(?<=[\.\s])[A-Ya-z](?=[\.\s])', ' ', x))
text['text'] = text['text'].apply(lambda x: re.sub('(?<![a-z])\.', ' ', x))
text['text'] = text['text'].apply(lambda x: " ".join(x.split()))

text = text[text['text'].notna()]['text'].tolist()
len(text)

MODEL_NAME = 'roberta'
epochs = 1
# Bert: Start: ('[CLS]', 101), End: ('[SEP]', 102), Mask: ('[MASK]',103), Pad: ('[PAD]', 0 )
# RoBERTa: Start: ('<s>', 0), End: ('</s>', 2), Mask: ('<mask>', 50264), Pad: ('<pad>', 1 )
# XLNet: Start: ('<cls>', 3), End: ('<sep>', 4), Mask: ('<mask>', 6), Pad: ('<pad>', 5 ) When encoding cls stands at the and of the previous token [24717, 4, 3]
# GPT-2: -


# # BERT
if MODEL_NAME == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bag = [item for sentence in text for item in sentence.split('.') if item != '']
    bag_size = len(bag)

    sentence_a = []
    sentence_b = []
    label = []

    for paragraph in text:
        sentences = [
            sentence for sentence in paragraph.split('.') if sentence != ''
        ]
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences - 2)

            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start + 1])
                label.append(0)
            else:
                index = random.randint(0, bag_size - 1)
                # this is NotNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(bag[index])
                label.append(1)

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

    # MLM
    # Copy Text from Next Sentence prediction
    inputs['labels'] = inputs.input_ids.detach().clone()

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array

    token_numbers = {'cls': 101, 'sep': 102, 'pad': 0, 'mask': 103}

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


# Prepare for RoBERTa
if MODEL_NAME == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    inputs['labels'] = inputs.input_ids.detach().clone()

    token_numbers = {'cls': 0, 'sep': 2, 'pad': 1, 'mask': 50264}

    rand = torch.rand(inputs['input_ids'].shape)
    mask_arr = (rand < 0.15) * (inputs['input_ids'] != token_numbers['cls']) * (inputs['input_ids'] != token_numbers['sep']) * (inputs['input_ids'] != token_numbers['pad'])

    selection = []
    # get indices of Trues (indices of to be masked words)
    for i in range(inputs['input_ids'].shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    # set value 103 instead of input_ids at inices
    for i in range(inputs['input_ids'].shape[0]):
        inputs['input_ids'][i, selection[i]] = token_numbers['mask']


# # Prepare for GPT-2

if MODEL_NAME == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    print(inputs)


if MODEL_NAME == 'xlnet':

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    mask_tokens_per_sequence = 20
    token_numbers = {'cls': 3, 'sep': 4, 'pad': 5, 'mask': 6}
    n_samples = inputs.input_ids.shape[0]
    sequence_length = inputs.input_ids.shape[1]  # inputs.input_ids.shape[1]

    # initialize perm_masks and target_mapping with zeros
    inputs['perm_mask'] = torch.zeros((n_samples, sequence_length, sequence_length), dtype=torch.float)
    inputs['target_mapping'] = torch.zeros((n_samples, mask_tokens_per_sequence, sequence_length), dtype=torch.float)

    # randomly select same amount of tokens per sequence
    masks = torch.tensor([sorted(random.sample(range(1, 510), mask_tokens_per_sequence)) for x in range(n_samples)])

    # set randomly selected tokens to 1 in perm_mask that all other words cannot see these tokens
    for i in range(n_samples):
        inputs['perm_mask'][i, :, masks[i]] = 1.0

    # set randomly selected tokens in accending order to target_mapping that they will be predicted one by one
    for i, i_mask in enumerate(masks):
        for j in range(mask_tokens_per_sequence):
            inputs['target_mapping'][i, j, i_mask[j]] = 1.0

    labels = []
    for i in range(n_samples):
        labels.append(inputs.input_ids[i, masks[i]].tolist())
    inputs['labels'] = torch.tensor(labels)

    print('input_ids: ', inputs['input_ids'].shape)
    print('perm_mask: ', inputs['perm_mask'].shape)
    print('target_mapping: ', inputs['target_mapping'].shape)
    print('label: ', inputs['labels'].shape)


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = OurDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

# Model
if MODEL_NAME == 'bert':
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
if MODEL_NAME == 'roberta':
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
if MODEL_NAME == 'xlnet':
    model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
if MODEL_NAME == 'gpt2':
    model = GPT2LMHeadModel.from_pretrained("gpt2")

model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)



for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)

        # process
        if MODEL_NAME == 'roberta':
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        elif MODEL_NAME == 'bert':
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids,  # input_ids with masked: 103
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,  # indication 0: sentence A, 1: sentence B
                next_sentence_label=next_sentence_label,  # indication 1: stentence B follows sentence A
                labels=labels  # all input_ids without masked words
            )

        elif MODEL_NAME == 'xlnet':
            perm_mask = batch['perm_mask'].to(device)
            target_mapping = batch['target_mapping'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                labels=labels
            )

        elif MODEL_NAME == 'gpt2':
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(
                input_ids=attention_mask,
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

model.save_pretrained(f'training/text/transformers/models/pretrained_{MODEL_NAME}/')
