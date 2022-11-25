from torch import nn
import copy


class BERTClassifier(nn.Module):

  def __init__(self, transformer_layer, n_classes):
    super(BERTClassifier, self).__init__()
    self.transformer = copy.deepcopy(transformer_layer)
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


class XLNetClassifier(nn.Module):
  def __init__(self, transformer_layer, n_classes):
    super(XLNetClassifier, self).__init__()
    self.transformer = copy.deepcopy(transformer_layer)
    self.drop = nn.Dropout(p=0.1)
    self.linear = nn.Linear(in_features=self.transformer.config.hidden_size * 512, out_features=n_classes)

  def forward(self, input_ids, attention_mask):

    output = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    ).last_hidden_state
    batch_size = output.shape[0]
    output = output.view(batch_size, -1)

    output = self.drop(output)
    output = self.linear(output)

    return output


class GPT2Classifier(nn.Module):
  def __init__(self, transformer_layer, n_classes):
    super(GPT2Classifier, self).__init__()
    self.transformer = copy.deepcopy(transformer_layer)
    self.drop = nn.Dropout(p=0.1)
    self.linear = nn.Linear(in_features=self.transformer.config.hidden_size * 512, out_features=n_classes)

  def forward(self, input_ids, attention_mask):

    output = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    ).last_hidden_state
    batch_size = output.shape[0]
    output = output.view(batch_size, -1)

    output = self.drop(output)
    output = self.linear(output)

    return output
