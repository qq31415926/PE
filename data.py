import os

import torch as th
import torch.nn as nn
import torch.utils as tutils
from torch.utils.data import TensorDataset, DataLoader
# import torchtext as thtext
from transformers import BertTokenizer, BertModel

import numpy as np

def clean(sentences):
    sentence1 = []
    for sentence in sentences:
        sentence = sentence.split()
        sentence_clean = [ t for t in sentence if t != '.' and t != ',' and t !=  "'" and t != "?" and t != "-"]
        sentence_clean = " ".join(sentence_clean)
        sentence1.append(sentence_clean)
    return sentence1
def preprocess(text, tokenizer):
    # text_ipt = tokenizer(
    #     text, padding=True, truncation=True, max_length=64, return_tensors="pt",
    # )
    # text_ipt = {
    #     "input_ids" : 
    # }
    input_ids = []
    token_ids = []
    attn_masks = []
    max_len = 64
    for sen in text:
        input = tokenizer.tokenize(sen)
        # toks = tokenizer.convert_tokens_to_ids(toks)
        tokids = tokenizer.convert_tokens_to_ids(input)
        mask = [1] * len(tokids)
        # assert len(input) == len(mask)
        if len(input) <= max_len - 2:
            tokids = [101] + tokids + [102]
            mask = [1] + mask + [1]
            if len(tokids) <= max_len:
                tokids = tokids + [0] * (max_len - len(tokids))
                mask = mask + [0] * (max_len - len(mask))
        else:
            tokids = [101] + tokids[:max_len - 2] + [102]
            mask = [1] + mask[:max_len - 2] + [1]
        token_type = [1] * max_len
        assert len(tokids) == max_len and len(mask) == max_len and len(token_type) == max_len
        input_id = np.array(tokids)
        token_id = np.array(token_type)
        attn_mask = np.array(mask)
        input_ids.append(input_id)
        token_ids.append(token_id)
        attn_masks.append(attn_mask)
    text_ipt = \
    {
        "input_ids":th.from_numpy(np.array(input_ids)).to(dtype=th.long),
        "token_type_ids":th.from_numpy(np.array(token_ids)).to(dtype=th.long),
        "attention_mask":th.from_numpy(np.array(attn_masks)).to(dtype=th.long)
    }
    return text_ipt


if __name__ == "__main__":
    bert_pretrained_file = "/public1/home/stu52265901009/cq/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_file, do_lower_case=True)

    data_path = "./data/tokens"
    # os.path.
    # neg_file = os.path.join(data_path, "neg")
    # pos_file = os.path.join(data_path, "pos")
    # neg_files = os.listdir(neg_file)
    # pos_files = os.listdir(pos_file)
    # print(len(neg_files),len(pos_files))
    neg_file = "./data/neg"
    pos_file = "./data/pos"
    neg_text, pos_text = [], []
    with open(neg_file,"r",encoding="utf-8") as f:
        neg_text = f.readlines()
    with open(pos_file,"r",encoding="utf-8") as f:
        pos_text = f.readlines()
    # for neg_file_ in neg_files:
    #     neg_file_ = "/public1/home/stu52265901009/cq/PoincareProbe-main/SentimentProbe/data/tokens/neg/"+neg_file_
    #     with open(neg_file_, "r", encoding="latin-1") as f:
    #         for line in f:
    #             neg_text.append(line.strip())
    # for pos_file_ in pos_files:
    #     pos_file_ = "/public1/home/stu52265901009/cq/PoincareProbe-main/SentimentProbe/data/tokens/pos/" + pos_file_ 
    #     with open(pos_file_, "r", encoding="latin-1") as f:
    #         for line in f:
    #             pos_text.append(line.strip())
    pos_text = clean(pos_text)
    neg_text = clean(neg_text)
    # with open("test.tsv","w") as fw:
    #     for pos 
    neg_len = [len(text.split()) for text in neg_text]
    pos_len = [len(text.split()) for text in pos_text]

    print(f"Average length: {np.mean(neg_len + pos_len):.2f}")
    print(f"Max length: {np.max(neg_len + pos_len)}")

    neg_text_ipt = preprocess(neg_text, tokenizer)
    neg_label = th.LongTensor([0] * len(neg_text))
    pos_text_ipt = preprocess(pos_text, tokenizer)
    pos_label = th.LongTensor([1] * len(pos_text))

    neg_dataset = TensorDataset(
        neg_text_ipt["input_ids"],
        neg_text_ipt["token_type_ids"],
        neg_text_ipt["attention_mask"],
        neg_label,
    )
    pos_dataset = TensorDataset(
        pos_text_ipt["input_ids"],
        pos_text_ipt["token_type_ids"],
        pos_text_ipt["attention_mask"],
        pos_label,
    )

    dataset = neg_dataset + pos_dataset
    dataset_len = len(dataset)
    print(dataset_len)
    train_dataset_len = 8528
    dev_dataset_len = 1067
    test_dataset_len = 1067

    train_dataset, dev_dataset, test_dataset = tutils.data.random_split(
        dataset, [train_dataset_len, dev_dataset_len, test_dataset_len]
    )

    print(f"Saving train/dev/test data to ./data")
    th.save(train_dataset, os.path.join("data", "train_datasetc.pt"))
    th.save(train_dataset, os.path.join("data", "dev_datasetc.pt"))
    th.save(test_dataset, os.path.join("data", "test_datasetc.pt"))
