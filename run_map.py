from copy import deepcopy
import torch as th
import torch
from transformers import BertModel,BertTokenizer
import argparse
from probe.probe import *
from encoder.bert_encoder1 import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import normalize 
import getopt as gt
from sytax.hyper import PoincareProbesytax
from util.cuda import get_max_available_gpu
import pickle
import torch.optim as optim
import torch.nn.functional as F
from typing import List


@torch.no_grad()
def calculate_difference(input_tensor,mask_tensor,all_mask,model,probs,tok_type_ids = None, order = 1):
    # non_padding_mask = ~padding_mask.unsqueeze(2) # B,L,1
    # input_tensor_no_padding = input_tensor.masked_fill(non_padding_mask, 0)
    B = input_tensor.size(0)
    L = input_tensor.size(1)
    label_ids = torch.argmax(probs,dim=-1) 
    probability_diffs = torch.zeros_like(input_tensor, dtype=torch.float32) if order == 1 else \
        torch.zeros(B,L,L)
    if order == 1:
        for i in range(B):
            for j in range(L):
                
                if mask_tensor[i][j].item() == 1:
                    row_idx, col_idx = i, j 
                    # print("input_tensor:{}".format(input_tensor[i]))
                    masked_input_tensor_copy = input_tensor[i].clone()
                    masked_input_tensor_copy[col_idx] = 0
                    masked_copy = all_mask[i].clone()
                    masked_copy[col_idx] = 0
                    if tok_type_ids is not None:
                        tok_type_ids_ = tok_type_ids[i]
                    if tok_type_ids is not None:
                        probs_m = model(masked_input_tensor_copy.view(1,-1),masked_copy.view(1,-1), token_type_ids = tok_type_ids_.view(1,-1))
                    else:
                        probs_m = model(masked_input_tensor_copy.view(1,-1),masked_copy.view(1,-1))
                    label_id = label_ids[i].item()
                    probability_diffs[row_idx, col_idx] = probs[i][label_id] - probs_m[0][label_id]
    elif order == 2:
        for i in range(B):
            for j in range(L):
                for k in range(j+1,L):
                    if mask_tensor[i][j].item() == 1 and mask_tensor[i][k].item() == 1:
                        row_idx, col_idx1, col_idx2 = i, j, k
                        masked_input_tensor_copy = input_tensor[i].clone()
                        masked_input_tensor_copy[col_idx1] = 0
                        masked_input_tensor_copy[col_idx2] = 0

                        
                        masked_copy = all_mask[i].clone()
                        masked_copy[col_idx1] = 0
                        masked_copy[col_idx2] = 0
                        probs_m = model(masked_input_tensor_copy.view(1,-1),masked_copy.view(1,-1))
                        label_id = label_ids[i].item()
                        probability_diffs[row_idx, col_idx1, col_idx2] = probs[i][label_id] - probs_m[0][label_id]

    return probability_diffs
    # for i in range(len(nonzero_positions[0])):
    #     row_idx, col_idx = nonzero_positions[0][i].item(), nonzero_positions[1][i].item()
    #     masked_input_tensor_copy = input_tensor.clone()
    #     masked_input_tensor_copy[row_idx, col_idx] = 0
    #     masked_copy = all_mask.clone()
    #     masked_copy[row_idx, col_idx] = 0
    #     probs_m = model(masked_input_tensor_copy,masked_copy)
    #     probability_diffs[row_idx, col_idx] = probs_m
        


    # mask_after = torch.cat([mask_tensor[:, 1:], torch.zeros_like(mask_tensor[:, -1:])], dim=1)
    # mask = (mask_tensor - mask_after).clamp(min = 0)
    # mask = ~mask 
    # input_tensor = input_tensor.masked_fill(mask, 0)
    # print(mask)
    # input_tensor = input_tensor * mask
    # prob_mask = model(input_ids = input_tensor, attention_mask = mask)
    # return probs - prob_mask

def make_key(texts:List[List[int]]):
    keys = ["_".join([str(t) for t in text]) for text in texts]
    return keys

if __name__ == "__main__":
    
    
    argp = argparse.ArgumentParser()
    argp.add_argument("--save", type=bool, default=True, help="Save probe")
    argp.add_argument("--cuda", type=int, help="CUDA device")
    argp.add_argument("--bert_path",type=str,default="")
    argp.add_argument("--num_classes",default=2,type=int,help="number of classes")
    argp.add_argument("--hidden_dropout_prob",type=float,default=0.1)
    argp.add_argument("--hidden_size",type=int,default=768)
    argp.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    argp.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    argp.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    argp.add_argument("--poincare_layernum",default=11)
    argp.add_argument("--poincare_dim",default=64)
    argp.add_argument("--alpha",type=float,default=1)
    argp.add_argument("--batch_size",type=int,default=32)
    argp.add_argument("--delta_lr",type=float,default=1e-2)
    argp.add_argument("--train_epochs",type=int,default=5)
    argp.add_argument("--eval_step",type=int,default=50)
    argp.add_argument("--eval_batch_size",type=int,default=64)
    argp.add_argument("--dataset",type=str,default="ptb")
    argp.add_argument("--order",type=int,default=1)

    args = argp.parse_args()

    bert_pretrained_file = args.bert_path
    vocab_file = os.path.join(bert_pretrained_file,"vocab.txt")
    with open(vocab_file,"r") as f:
        vocabs = f.readlines()
    vocab = [tok.strip() for tok in vocabs]
    id2tok = {ID : tok for ID,tok in enumerate(vocab)}
    tok2id = {tok : ID for ID,tok in enumerate(vocab)} 
    bert = BertModel.from_pretrained(bert_pretrained_file)

    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_file)



    # args = parser.parse_args()
    if args.cuda is not None:
        device_id = args.cuda
    else:
        device_id, _ = get_max_available_gpu()
    device = th.device("cuda:" + str(device_id) if th.cuda.is_available() else "cpu")
    # if args.ckpt != None:
    #     bert = torch.load(ckpt)
    model = bertEncoder(args)
    # torch.load
    # print(type(model))
    if args.dataset == "ptb":
        ckpt = "./checkpoint/bestmodel.pt"
        data_path = "./data"
        with open(ckpt,"rb") as f:
            model_state_dict = torch.load(f)
            model.load_state_dict(model_state_dict)
    elif args.dataset == "snli":
        ckpt = "./checkpoint/snli/state.ckpt"
        data_path = "./data/snli"
        with open(ckpt, "rb") as f:
            ckpt = pickle.load(f)
            del ckpt["bertmodel.embeddings.position_ids"]
            model.load_state_dict(ckpt)
    elif args.dataset == "trec":
        data_path = './data/trec'
        output_dir = "./checkpoint/"
        savename = "./checkpoint/trec/bestmodel.pt"
    elif args.dataset == "yelp":
        data_path = './data/yelp'
        output_dir = "./checkpoint/"
        savename = "./checkpoint/yelp/bestmodel.pt"
    for param in model.parameters():
        param.requires_grad = False
    for param in bert.parameters():
        param.requires_grad = False
    model.to(device)
    bert.to(device)

    # train_dataset = th.load(os.path.join(data_path, "train_dataset.pt"))
    # dev_dataset = th.load(os.path.join(data_path, "dev_dataset.pt"))
    test_dataset = th.load(os.path.join(data_path, "test_dataset.pt"))

    # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # dev_data_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size = args.eval_batch_size, shuffle=False)

    
    test_id = 0
    punctions = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']

    forbidden_tok = [tok2id[t] for t in punctions]
   
       
           
    pad_token = [tok2id["[CLS]"], tok2id["[SEP]"], tok2id["[PAD]"]]
    ground_truth = []
    best_loss = -1
    bid = 0
    aopc_map = {}
    for batch in tqdm(test_data_loader, desc="[Constructing Train]"):
        bid += 1
        if args.dataset == "ptb":
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
        elif args.dataset == "trec" or args.dataset == "yelp":
            text_input_ids, text_attention_mask, text_token_type_ids, label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
        elif args.dataset == "snli":
            text_input_ids, text_token_type_ids, text_attention_mask,  label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(device),
            text_token_type_ids.to(device),
            text_attention_mask.to(device),
            label.to(device),
        )
        text1 = batch[0].cpu().numpy().tolist()
        B = len(text1)
        L = len(text1[0])
        new_text1 = [[t for t in text1[i] if t not in forbidden_tok] for i in range(B)]
        # raw_text1 = [[t for t in new_text1[i] if t not in pad_token] for i in range(B)]
        # raw_mask = 
        new_text1 = [x + [0] * (L - len(x)) if len(x) < L else x for x in new_text1 ]
        hash_key = make_key(new_text1)
        raw_mask = [[1 if t not in pad_token else 0 for t in x] for x in new_text1]
        mask1 = [[1 if t != 0 else 0 for t in x ] for x in new_text1]
        
        new_text1 = torch.LongTensor(new_text1).to(device)
        mask1 = torch.LongTensor(mask1).to(device)
        raw_mask = torch.LongTensor(raw_mask).to(device)
        if args.dataset == "ptb" or args.dataset == "trec" or args.dataset == "yelp":
            probs,_ = model(new_text1,attention_mask = mask1,output_hidden_states=True)
            target = calculate_difference(input_tensor=new_text1,mask_tensor=raw_mask,all_mask = mask1,model=model,probs=probs,order=args.order)
        elif args.dataset == "snli":
            probs,_ = model(input_ids = new_text1, attention_mask = mask1, token_type_ids = text_token_type_ids, output_hidden_states = True)
            target = calculate_difference(input_tensor=new_text1,mask_tensor=raw_mask,all_mask = mask1,model=model,probs=probs, tok_type_ids = text_token_type_ids)
        target = target.cpu().numpy().tolist()
        # if args.order == 1:
        for key , aopc_list in zip(hash_key, target):
            aopc_map[key] = aopc_list
        # elif args.order == 2:
    save_path = "./first_order/{}/test/aopc_map.pkl".format(args.dataset) if args.order == 1 else \
        f"./second_order/{args.dataset}/test/aopc_map.pkl"
    # if args.dataset == "ptb":
    print(save_path)
    with open(save_path,"wb") as fw:
        pickle.dump(aopc_map,fw)
    