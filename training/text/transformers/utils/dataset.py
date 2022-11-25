"""Initialize TextDataset."""
#
from .helper import MAX_LEN

# torch
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
  """Dataset Class."""

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
