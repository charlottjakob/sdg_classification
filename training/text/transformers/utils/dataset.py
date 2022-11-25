"""Initialize TextDataset."""

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
    text = str(self.texts[idx])
    target = self.targets[idx]
    encoding = self.tokenizer(
        text,
        return_tensors='pt',
        max_length=512, 
        truncation=True, 
        padding='max_length'     
    )
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
