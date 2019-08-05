'''
This program is for encoding texts for each bird.
'''

import torch
import configparser
from tqdm import tqdm
import numpy as np
from pytorch_transformers import BertConfig,BertTokenizer, BertModel, BertForMaskedLM

def texts_encoder(cfg_path):
    '''
    args:
        cfg_path : path to cfg file.
    return:
        (bool) : successed or failed
    save:
        (numpy array) : texts encoding, sized [number of data, 1, 768]
    '''
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    path_text_folder = cfg["Dataset"]["PATH_TEXT_FOLDER"]
    path_text_txt = cfg["Dataset"]["PATH_TEXT_TXT"]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    model = model.to(device)

    texts = []

    with open(path_text_txt) as f_text:
        lines_text = f_text.readlines()

    for idx, line_text in enumerate(tqdm(lines_text)):
        # text loading
        splited = line_text.strip().split()
        text_idx = splited[0]
        text = line_text.strip().replace(text_idx+' ','')
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0 for i in range(len(indexed_tokens))]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            texts.append(outputs[1].cpu().numpy())
    texts = np.array(texts)

    if not os.path.exists("./data"):
        os.makedirs("./data")
    np.save('./data/texts_encoded.npy',texts)
    return True


def main():
    cfg_path = './config/GAWWN_v1.cfg'
    if texts_encoder(cfg_path):
        print("Bert encoding succeed!")
    else:
        print("ERROR: Can't not encode texts using Bert.")
    return 0


if __name__ == '__main__':
    main()
