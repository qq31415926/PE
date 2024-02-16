import torch as th

import numpy as np
from tqdm import tqdm
from copy import deepcopy
# from bert_encoder import *
import random
from typing import List
from itertools import combinations
type_num = {"ptb": 2, "snli": 3, "trec": 6, "yelp" : 2}
def train(
    train_data_loader,
    probe,
    bert,
    loss_fct,
    optimizer,
    dev_data_loader=None,
    scheduler=None,
):
    # Train the probe
    probe.train()
    train_loss, dev_loss = 0, 0
    train_acc, dev_acc = 0, 0
    for batch in tqdm(train_data_loader, desc="[Train]"):
        optimizer.zero_grad()

        # print("batch0:{} batch1:{} batch2:{} batch3:{}".format(batch[0], batch[1], batch[2], batch[3]))
        text_input_ids,  text_attention_mask, text_token_type_ids, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(probe.device),
            text_token_type_ids.to(probe.device),
            text_attention_mask.to(probe.device),
            label.to(probe.device),
        )

        with th.no_grad():
            outputs = bert(
                text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs[2] # 12 layers output
            # 10th layer 
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )
        # print(sequence_output.shape)
        logits = probe(sequence_output)
        # print(logits.shape)
        if probe.type == "trec":
            logits = logits.unsqueeze(0)
        C = type_num[probe.type]
        # print(C)
        # print("logits shape:{} C:{} label shape:{}".format(logits.shape, C, label.shape))
        # print("logits:{} label:{}".format(logits, label))
        l = loss_fct(logits.view(-1, C), label.view(-1))


        train_loss += l.item()
        l.backward()
        optimizer.step()

        train_acc += (logits.argmax(-1) == label).sum().item()
    train_loss = train_loss / len(train_data_loader.dataset)
    train_acc = train_acc / len(train_data_loader.dataset)

    if dev_data_loader is not None:
        probe.eval()
        for batch in tqdm(dev_data_loader, desc="[Dev]"):
            text_input_ids,text_attention_mask, text_token_type_ids,  label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                text_input_ids.to(probe.device),
                text_token_type_ids.to(probe.device),
                text_attention_mask.to(probe.device),
                label.to(probe.device),
            )
            with th.no_grad():
                outputs = bert(
                    text_input_ids,
                    attention_mask=text_attention_mask,
                    token_type_ids=text_token_type_ids,
                    output_hidden_states=True,
                )
                hidden_states = outputs[2]
                sequence_output = (
                    hidden_states[probe.layer_num]
                    .to(probe.device)
                    .to(probe.default_dtype)
                )
                logits = probe(sequence_output)

                C = type_num[probe.type]
                l = loss_fct(logits.view(-1, C), label.view(-1))
                dev_loss += l.item()

            dev_acc += (logits.argmax(-1) == label).sum().item()
        # Adjust the learning rate
        if scheduler is not None:
            scheduler.step(dev_loss)

        dev_loss = dev_loss / len(dev_data_loader.dataset)
        dev_acc = dev_acc / len(dev_data_loader.dataset)

    return (
        train_loss,
        train_acc,
        dev_loss,
        dev_acc,
    )

def remove_puncture(batch,forbidden_tok,pad_token):
    text1 = batch[0].cpu().numpy().tolist()
    B = len(text1)
    L = len(text1[0])
    new_text1 = [[t for t in text1[i] if t not in forbidden_tok] for i in range(B)]
    # raw_text1 = [[t for t in new_text1[i] if t not in pad_token] for i in range(B)]
    # raw_mask = 
    new_text1 = [x + [0] * (L - len(x)) if len(x) < L else x for x in new_text1 ]
    # hash_keys = make_key(new_text1)
    raw_mask = [[1 if t not in pad_token else 0 for t in x] for x in new_text1]
    # raw_mask_ = deepcopy(raw_mask)
    mask1 = [[1 if t != 0 else 0 for t in x ] for x in new_text1]
    return new_text1, raw_mask, mask1
@th.no_grad
def construct_pair_test(raw_mask_, hidden_states_cpu, model, device):
    target_ind_no_pad = [[x2 for x2 in range(len(x1)) if x1[x2] != 0 ] for x1 in raw_mask_]
    pair_tok = [list(combinations(t, 2)) for t in target_ind_no_pad]
    
    pair_input = [
        [
            np.concatenate([ hidden_states_cpu[b][p[0]], hidden_states_cpu[b][p[1]]])
            for p in sentence_pair
        ]
        
        for b, sentence_pair in zip(range(len(hidden_states_cpu)), pair_tok)
    ]
    pair_input_pt = th.tensor(pair_input).to(device)
    output = model(pair_input_pt).squeeze()
    sigmoid = th.nn.Sigmoid()
    logits = sigmoid(output)
    logits = (logits > 0).to(th.int)
    return logits.cpu().numpy().tolist(), pair_tok

def construct_pair(args, raw_mask_, hidden_states_cpu, target):
    target_ind_no_pad = [[x2 for x2 in range(len(x1)) if x2 != 0 ] for x1 in raw_mask_]

    uniform_candidate_tok = [random.sample(t, int(args.sample_ratio * len(t))) for t in target_ind_no_pad]

    max_pairL = int(args.sample_ratio * args.maxL)
    uniform_candidate_tok = [x + [args.maxL - 1] * (max_pairL - len(x)) if len(x) < max_pairL else x for x in uniform_candidate_tok]

    pair_tok = [
           list(combinations(candidate,2)) for candidate in uniform_candidate_tok
        ]
    # print("pair_tok:{}".format(pair_tok))
    pair_ground_truth = [
        [
            int(target_[p[0]] > target_[p[1]]) 
            for p in candidate
        ] 
        for candidate, target_ in zip(pair_tok, target)
        
    ]
    # print("hidden_states:{}".format(hidden_states_cpu.shape))
    # for b in range(len(hidden_states_cpu)):
    #     for pair in pair_tok:
    #         print("hidden_states_cpu[b][pair[0]]:{} hidden_states_cpu[b][pair[1]]:{}".format(hidden_states_cpu[b][pair[0]].shape, hidden_states_cpu[b][pair[1]].shape))
    #         exit
    # for pair in pair_tok:
    #     print("pair:{} pair[0]:{} pair[1]:{}".format(pair, pair[0],pair[1]))
    
    pair_input = [
        [
            np.concatenate([ hidden_states_cpu[b][p[0]], hidden_states_cpu[b][p[1]]])
            for p in sentence_pair
        ]
        
        for b, sentence_pair in zip(range(len(hidden_states_cpu)), pair_tok)
    ]

            # pair_ground_truth = [[]]
    # B,Combination(max_pairL,2)
    # pair_ground_truth = [[int(target_[pair[0]] > target_[pair[1]]) for sentence_pair in pair_tok] for target_ in target]
        
    # B,Combination(max_pairL,2),D
    # pair_input = [[np.concatenate(hidden_states_cpu[b][pair[0]], hidden_states_cpu[b][pair[1]]) for pair in pair_tok] for b in range(len(hidden_states_cpu))]

    return pair_input, pair_ground_truth
def make_key(texts:List[List[int]]):
    keys = ["_".join([str(t) for t in text]) for text in texts]
    return keys
def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return th.softmax(logits, dim=-1)
def construct_graph_weight(logits, pair_tok, raw_mask_):
    # print("logits:{} pair_tok:{} raw_mask_:{}".format(logits, pair_tok, raw_mask_))
    # logits = logits[0]
    if type(logits) == int:
        logits = [logits] 
    pair_tok = pair_tok[0]
    raw_mask_ = raw_mask_[0]
    # node_num = len(logits)
    # graph = [-1] * maxL
    node_weight = [0 if raw_mask_x == 1 else -1 for raw_mask_x in raw_mask_]


    for y, p in zip(logits, pair_tok):
        if y == 1:
            node_weight[p[0]] += 1
        elif y == 0:
            node_weight[p[1]] += 1
    return node_weight