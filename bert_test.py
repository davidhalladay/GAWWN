# test the Bert word embedding

# BERT example

import torch
from pytorch_transformers import BertConfig,BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] There is a red bird"
tokenized_text = tokenizer.tokenize(text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0, 0, 0, 0, 0, 0]

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs[1]
    print(encoded_layers.min())
    print(encoded_layers.max())
    print(encoded_layers.shape)
