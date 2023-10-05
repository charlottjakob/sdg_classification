from .utils import get_default_device
from training import BERTClassifier

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
from torch import nn
from training import BERTClassifier



class AttentionClass():

    def __init__(self, sdg_idx) -> None:
        
        self.sdg_idx = sdg_idx
        self.device = get_default_device()
        
        from training import BERTClassifier
        model = torch.load(f'/Users/charlott/Documents/github_repos/sdg_classification/models/bert.model', map_location=self.device)
        model.eval()
        model.zero_grad()

        self.model = model
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        


    def construct_input_ref_pair(self, text, ref_token_id, sep_token_id, cls_token_id):

        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # construct input token ids
        input_ids = [cls_token_id] + text_ids + [sep_token_id]

        # construct reference token ids 
        ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

        return torch.tensor([input_ids], device=self.device),  torch.tensor([ref_input_ids], device=self.device), len(text_ids)

    def construct_input_ref_token_type_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=self.device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device)# * -1
        return token_type_ids, ref_token_type_ids

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids
        
    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def predict(self, input_ids):
        return self.model(input_ids, None)[0] # remove the first dimension

    def custom_forward(self,inputs):
        outputs = self.predict(inputs)
        return torch.sigmoid(outputs)[self.sdg_idx].unsqueeze(-1)
    

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions



    def run_attention_extraction(self, text="", sdg_idx=0):


        ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference
        sep_token_id = self.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        cls_token_id = self.tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


        lig = LayerIntegratedGradients(self.custom_forward, self.model.transformer.embeddings)


        input_ids, ref_input_ids, sep_id = self.construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
        token_type_ids, ref_token_type_ids = self.construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = self.tokenizer.convert_ids_to_tokens(indices)


        # show influence towards the prediction: 
        # if more positive, the word is more important for the SDG
        # if more negative, the wordd even has a bad influence on the SDG
        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=ref_input_ids,
                                            n_steps=50,
                                            internal_batch_size=3,
                                            return_convergence_delta=True)


        # attributions size: [additional dimension for torch,  number of words/input_ids, hidden size of BERT-Layer]
        print("attributions size: ", attributions.size())
        attributions_sum = self.summarize_attributions(attributions)
        
        
        # TODO: prediction only once for the same datapoint but different SDGs
        scores = self.predict(input_ids)
        
        
        return all_tokens, attributions_sum, delta, scores



