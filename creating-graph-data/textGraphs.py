## importing libraries
import os

import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch import nn


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #removes links
    text = re.sub(r'(?P<url>https?://[^\s]+)', r'', text)

    # remove @usernames
    text = re.sub(r"\@(\w+)", "", text)

    #remove # from #tags
    text = text.replace('#', '')

    return text


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        ## Main DataFrame with all the tweets
        self.df = df.reset_index(drop=True)

        ## TOkenizer to be used
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):# 将张量转换为python标量int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## Textual content of the post
        text = self.df['tweetText'][idx]

        ## Unique id to be used as identifier
        unique_id = self.df['tweetId'][idx]

        encoded_sent = self.tokenizer.encode_plus(
            text=text_preprocessing(text),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=512,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            return_attention_mask=True,  # Return attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'unique_id': unique_id}


def store_data(bert, device, df, dataset, store_dir):# 把每条推文用bert编码后，保存成两份.npy文件
    lengths = []
    bert.eval()

    for idx in tqdm(range(len(df))):
        sample = dataset.__getitem__(idx)

        input_ids, attention_mask = sample['input_ids'].unsqueeze(0), sample['attention_mask'].unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        unique_id = sample['unique_id']

        # .sum() 得到真实 token 个数（包含 [CLS] 与 [SEP]）
        # .detach().cpu().item()把标量张量 → Python int。
        num_tokens = attention_mask.sum().detach().cpu().item()

        with torch.no_grad():
            out = bert(input_ids=input_ids, attention_mask=attention_mask)

        # 去掉开头的 [CLS] 和结尾的 [SEP]；只保留真实文本对应的那些 token。
        out_tokens = out.last_hidden_state[:, 1:num_tokens, :].detach().cpu().squeeze(0).numpy()  ## token vectors

        ## Save token-level representations
        filename = f'{root_dir}{store_dir}{unique_id}.npy'
        np.save(filename, out_tokens)

        lengths.append(num_tokens)

        ## Save semantic/ whole text representation
        out_cls = out.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().squeeze(0).numpy()  ## cls vector
        filename = f'{root_dir}{store_dir}{unique_id}_full_text.npy'
        np.save(filename, out_cls)

    return lengths


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Base directory for the data
    root_dir = "E:/GAME-ON-main/creating-graph-data/dataset/twitter/"

    ## File locations
    train_csv_name = "train.csv"
    test_csv_name = "test.csv"

    ## Loading model and tokenizer from HuggingFace
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True) ## ## bert-base-uncased - for english dataset
    # tokenizer = BertTokenizer.from_pretrained(
    #     'E:/GAME-ON-main/pretrained/bert-base-chinese/',
    #     do_lower_case=True
    # )
    # bert = BertModel.from_pretrained('bert-base-chinese',
    #                     return_dict=True)
    local_path = 'E:/GAME-ON-main/pretrained/bert-base-chinese/'
    tokenizer = BertTokenizer.from_pretrained(local_path, do_lower_case=True)# 转换为小写
    bert = BertModel.from_pretrained(local_path)

    bert = bert.to(device)

    ## Directory to store the node embeddings for each post
    store_dir = 'bert_tokens/'

    ## Create graph data for training set
    df_train = pd.read_csv(os.path.join(root_dir, train_csv_name), sep='\t')
    df_train = df_train.dropna().reset_index(drop=True) # 把任何列里只要出现 NaN 的行整行删除，重新生成索引
    train_dataset = TextDataset(df_train, tokenizer)

    lengths = store_data(bert, device, df_train, train_dataset, store_dir)

    ## Create graph data for testing set
    df_test = pd.read_csv(os.path.join(root_dir, test_csv_name), sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    test_dataset = TextDataset(df_test, tokenizer)

    lengths = store_data(bert, device, df_test, test_dataset, store_dir)
