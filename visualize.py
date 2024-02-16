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
from util.lca import hyp_lca
from sytax.hyper import PoincareProbesytax
from util.cuda import get_max_available_gpu
from util.mst import get_mst, EdgeNode, calculate_level
from util.train import remove_puncture,  make_key

# from residual import deltaNet
from util.normalize import min_max_normalize
from util.AOPC import calculate_metric
from utils_glue import _truncate_seq_pair

import pickle
import heapq
import time
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    
    
    
    argp = argparse.ArgumentParser()
    argp.add_argument("--save", type=bool, default=True, help="Save probe")
    argp.add_argument("--cuda", type=int, help="CUDA device")
    argp.add_argument("--bert_path",type=str,default="")
    argp.add_argument("--num_classes",default = 3,type=int,help="number of classes")
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
    argp.add_argument("--alpha1",type=float,default=0)
    argp.add_argument("--alpha2",type=float,default=0)
    argp.add_argument("--alpha3",type=float,default=1)
    argp.add_argument("--top",type=float,default=0.2)
    argp.add_argument("--del_or_pad",type=str,default="del")
    argp.add_argument("--maxL",type=int,default=64)
    argp.add_argument("--type",default="pairwise")
    argp.add_argument("--normalize",default="minmax")
    argp.add_argument("--use_delta",action="store_true")
    argp.add_argument("--pool_size",type=int,default=1)
    argp.add_argument("--dataset",type=str, default="ptb")


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
    if args.dataset == "ptb":
        ckpt = "./checkpoint/bestmodel.pt"
        data_path = "./data"
        args.num_classes = 2
    elif args.dataset == "snli":
        ckpt = "./checkpoint/snli/state.ckpt"
        data_path = "./data/snli"
        args.num_classes = 3
    elif args.dataset == "trec":
        ckpt = "./checkpoint/trec/bestmodel.pt"
        data_path = "./data/trec"
        args.num_classes = 6
    elif args.dataset == "yelp":
        ckpt = "./checkpoint/yelp/bestmodel.pt"
        data_path = "./data/yelp"
        args.num_classes = 2
    
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
    if args.dataset == "snli":
        with open(ckpt,"rb") as f:
            # model_state_dict = torch.load(f)
            ckpt = pickle.load(f)
            del ckpt["bertmodel.embeddings.position_ids"]
            model.load_state_dict(ckpt)
    elif args.dataset == "ptb" or args.dataset == "trec" or args.dataset == "yelp":
        with open(ckpt,"rb") as f:
            ckpt = torch.load(f)
            model.load_state_dict(ckpt)
    model.to(device)
    bert.to(device)

    if args.dataset == "ptb":
        poincare_path = "./checkpoint/sentiment-poincare-probe.pt"
    elif args.dataset == "snli":
        poincare_path = "./checkpoint/snli_layer=11_dim=64.pt"
    elif args.dataset == "trec":
        poincare_path = "./checkpoint/trec/layer=11_dim=64.pt"
    elif args.dataset == "yelp":
        poincare_path = "./checkpoint/yelp/layer=11_dim=64.pt"
    probe = PoincareProbe(
        device=device, default_dtype=torch.float64, layer_num=args.poincare_layernum, type = args.dataset
    )
    with open(poincare_path,"rb") as f:
        ckpt_poincare = torch.load(f,map_location=device)
        probe.load_state_dict(ckpt_poincare)
    probe.to(device)

    sytax_probe = PoincareProbesytax(device=device,curvature=-1,dim_hidden=64,dim_in=768,dim_out=64)
    sytax_ckpt = "./checkpoint/syntax-poincare-probe.pt"
    sytax_probe.to(device)
    with open(sytax_ckpt,"rb") as f:
        sytax_st = torch.load(f,map_location=device)
        sytax_probe.load_state_dict(sytax_st)
    
    
    maxL = args.maxL
    def pred(input,mask,input_else=None, mask_else = None, tokenize = False):
        if tokenize:
            input = tokenizer.tokenize(" ".join(input))
            mask = [1] * len(input)
            if input_else is not None:
                input_else = tokenizer.tokenize(" ".join(input_else))
                mask_else = [1] * len(input_else)
        tokids = tokenizer.convert_tokens_to_ids(input)
        tokids_else = tokenizer.convert_tokens_to_ids(input_else) if input_else is not None else None
        # assert len(input) == len(mask)
        
        if args.dataset == "ptb" or input_else == None:
            if len(input) <= maxL - 2:
                tokids = [101] + tokids + [102]
                mask = [1] + mask + [1]
                if len(tokids) <= maxL:
                    tokids = tokids + [0] * (maxL - len(tokids))
                    mask = mask + [0] * (maxL - len(mask))
            else:
                tokids = [101] + tokids[:maxL - 2] + [102]
                mask = [1] + mask[:maxL - 2] + [1]
        elif args.dataset == "snli":
            toktype = []
            if len(input) + len(input_else) <= maxL - 3:
                tokids = [101] + tokids + [102] + tokids_else + [102]
                toktype = [0] + [0] * len(input) + [0] + [1] * len(input_else) + [1]
                mask = [1] + mask + [1] + mask_else + [1]
                if len(tokids) <= maxL:
                    tokids = tokids + [0] * (maxL - len(tokids))
                    mask = mask + [0] * (maxL - len(mask))
                    toktype = toktype + [1] * (maxL - len(toktype))
            else:
                _truncate_seq_pair(tokids, tokids_else,maxL - 3)
                a1 = len(tokids)
                a2 = len(tokids_else)
                tokids = [101] + tokids + [102] + tokids_else + [102]
                toktype = [0] + [0] * a1 + [0] + [1] * a2 + [1]
                mask = [1] + mask[:a1] + [1] + mask_else[:a2] + [1]


        # print(len(),len(mask))
        if args.dataset == "ptb" or input_else == None:
            assert len(tokids) == maxL and len(mask) == maxL
            input = torch.LongTensor(tokids).view(1,-1).to(device)
            mask = torch.LongTensor(mask).view(1,-1).to(device)
            with torch.no_grad():
                outputs = model(
                        input,
                        attention_mask=mask,
                    )
        elif args.dataset == "snli":
            assert len(tokids) == maxL and len(mask) == maxL and len(toktype) == maxL
            input = torch.LongTensor(tokids).view(1,-1).to(device)
            mask = torch.LongTensor(mask).view(1,-1).to(device)
            toktype = torch.LongTensor(toktype).view(1,-1).to(device)
            with torch.no_grad():
                outputs = model(
                        input,
                        attention_mask=mask,
                        token_type_ids=toktype
                    )
        
        return outputs
    
    
    test_dataset = th.load(os.path.join(data_path, "test_dataset.pt"))
   
    
    
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



    probe.eval()
    loss = 0
    acc = 0
    test_id = 0
    # aopcs = []
    punctions = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']

    forbidden_tok = [tok2id[t] for t in punctions]
    pad_token = [tok2id["[CLS]"], tok2id["[SEP]"], tok2id["[PAD]"]]
    # alpha1 = [0,1,2]
    alpha1 = [args.alpha1]
    # alpha2 = [0,1,2]
    alpha2 = [args.alpha2]
    alpha3 = [args.alpha3]
    aopcs = {}
    avg_time = []
    # if args.dataset == "ptb":
    with open(f"./second_order/{args.dataset}/test/aopc_map.pkl","rb") as f:
        aopc_map2 = pickle.load(f)
    with open(f"./first_order/{args.dataset}/test/aopc_map.pkl","rb") as f:
        aopc_map1 = pickle.load(f)
    heap = []
    # elif args.dataset == "snli":
    #     with open("./first_order/snli/test/aopc_map.pkl","rb") as f:
    #         aopc_map = pickle.load(f)
    # elif args.dataset == "trec":
    #     with open("./first_order/trec/test/aopc_map.pkl","rb") as f:
    #         aopc_map = pickle.load(f)
    for a1 in alpha1:
        for a2 in alpha2:
            for a3 in alpha3:
                aopcs[str(a1) + "_" + str(a2) + "_" + str(a3)] = []
    for batch in tqdm(test_data_loader, desc="[Evaluate]"):
    
        if args.dataset == "snli":
            text_input_ids, text_token_type_ids, text_attention_mask,  label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
        elif args.dataset == "ptb":
            text_input_ids, text_token_type_ids, text_attention_mask,  label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
        elif args.dataset == "trec" or args.dataset == "yelp":
            text_input_ids,text_attention_mask, text_token_type_ids,   label = (
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
        new_text1, raw_mask, mask1 =  remove_puncture(text_input_ids,forbidden_tok,pad_token)
        # if len(new_text1[0]) < 3:
        #     print(f"test_id:{test_id} Too Short!")
        #     test_id += 1
        #     continue
        new_text1_l = deepcopy(new_text1[0])
        text_key = make_key(new_text1)
        new_text1 = torch.LongTensor(new_text1).to(device)
        mask1 = torch.LongTensor(mask1).to(device)
        
        with th.no_grad():
            outputs_bert = bert(
                new_text1,
                attention_mask=mask1,
                token_type_ids=text_token_type_ids,
                output_hidden_states=True,
            )
            hidden_states_bert = outputs_bert[2]
            ## sytax
            # hidden_states_bert = hidden_states_bert[11]
            proj = sytax_probe.project(hidden_states_bert[11])
            depth = sytax_probe.depth(proj)
            # print("depth:{}".format(depth))
            # print(text_input_ids.shape,text_attention_mask.shape)
            probs,hidden_states = model(new_text1,attention_mask = mask1,output_hidden_states=True)
            label_id = torch.argmax(probs,dim=-1).item()
            # hidden_states = outputs[2]
            sequence_output = (
                hidden_states_bert[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )
            # probs = probe.forward(sequence_output) # 1,2
            # probs = normalize(probs)
            # label_id = torch.argmax(probs,dim = -1).item()
            logits = probe.forward_logits(sequence_output) # 1,L,d
            L = logits.size(1)
            # logits = normalize(logits,dim=-1)
            # sim = torch.matmul(logits,logits.T).view(L,-1) # 
            logits = logits.squeeze()
            if args.num_classes == 2:
                if label.item() == 0:
                    sim  = probe.ball.dist(probe.neg,logits)
                else:
                    sim = probe.ball.dist(probe.pos,logits)
            elif args.num_classes == 3:
                if label.item() == 0:
                    sim = probe.ball.dist(probe.c1, logits)
                elif label.item() == 1:
                    sim = probe.ball.dist(probe.c2, logits)
                else:
                    sim = probe.ball.dist(probe.c3, logits)
            elif args.num_classes == 6:
                sim = probe.ball.dist(probe.centriods[int(label.item())], logits)
            # sim = probe.ball.dist(logits,logits)
            # print(logits.shape)
            text_key = text_key[0]
            first_order = aopc_map1[text_key]
            second_order = aopc_map2[text_key]
            sim = sim.squeeze()
            depth = depth.squeeze()
            if args.normalize == "minmax":
                sim = min_max_normalize(sim)
                depth = min_max_normalize(depth)
                first_order = min_max_normalize(first_order)
                
            
            sim = sim.detach().cpu().numpy().tolist()
            # print("test_id:{} sim:{}".format(test_id, sim))
            depth = depth.detach().cpu().numpy().tolist()
            # print("test_id:{} depth:{}".format(test_id, depth)) 
            first_order = first_order.cpu().numpy().tolist()
            batch_text = new_text1.squeeze().cpu().numpy().tolist() # remove puncture

            batch_text = [id2tok[t] for t in batch_text]
            raw_text = [x for x in batch_text if x != "[PAD]" and x != "[CLS]" and x != "[SEP]"] # remove padding
            raw_text_cls_sep = [x for x in batch_text if x != "[PAD]"]
            raw_text_p = [(x, y) for x, y in zip(batch_text, range(len(batch_text))) if x != "[PAD]"]
            if len(raw_text_p) < 5:
                print(f"test_id:{test_id} Too Short!")
                test_id += 1
                continue
            # raw_text_no_pad = []
            # raw_text_no_pad_L = len(raw_text_p)
            # id2pid = {x[1] : y for x, y in zip(raw_text_p, range(len(raw_text_p)))} 
            # pid2id = {y : x for x, y in id2pid.items()}
            for a1 in alpha1:
                for a2 in alpha2:
                    for a3 in alpha3:
                        mst =  get_mst(a1,a2,a3,sim,depth,first_order,batch_text)
                        if args.dataset == "snli":
                            batch_text_a = batch_text[: batch_text.index("[SEP]")]
                            batch_text_b = batch_text[batch_text.index("[SEP]") + 1 :]
                            raw_text_a = [x for x in batch_text_a if x != "[PAD]" and x != "[CLS]" and x != "[SEP]"]
                            raw_text_b = [x for x in batch_text_b if x != "[PAD]" and x != "[CLS]" and x != "[SEP]"]
                            aopc = calculate_metric(args,pred,label_id,probs,mst,aopc_token=args.del_or_pad,pad_token="[PAD]",s_text=raw_text,metric="AOPC",s_text_a=raw_text_a,s_text_b=raw_text_b)
                        elif args.dataset == "ptb" or args.dataset == "trec" or args.dataset == "yelp":
                            aopc = calculate_metric(args,pred,label_id,probs,mst,aopc_token=args.del_or_pad,pad_token="[PAD]",s_text=raw_text,metric="AOPC")
                    
                        AOPC = np.array(aopc).mean()
                        key = str(a1) + "_" + str(a2) + "_" + str(a3)
                        aopcs[key].append(AOPC)
                        print("test_id:{} alpha1:{:.2f} alpha2:{:.2f} alpha3:{:.1f} AOPC:{:.3f} text:{} pred_label:{} ground_truth_label:{}".format(test_id,a1,a2,a3,AOPC,raw_text,label_id,label.item()))
            print("building tree")
            # if isinstance(second_order, torch.tensor):
            #     second_order 
            start_time = time.time()
            max_x, max_y, max_v = -1, -1, 0
            pad_len = new_text1.size(1)
            for tok1_id in range(pad_len):
                for tok2_id in range(tok1_id + 1, pad_len):
                    if second_order[tok1_id][tok2_id] > max_v and id2tok[new_text1[0,tok1_id].item()] != "[PAD]" and id2tok[new_text1[0,tok2_id].item()] != "[PAD]":
                        max_x = tok1_id
                        max_y = tok2_id
                        max_v = max_v
                        
                        # heapq.push(heap, EdgeNode(x = tok1_id, y = tok2_id, v = second_order[tok1_id][tok2_id] ))
            start_left = id2tok[new_text1[0,max_x].item()]
            start_right = id2tok[new_text1[0,max_y].item()]
            print(f"test_id:{test_id} Level:0 node:{start_left} node:{start_right} ")
            
            no_pad_L = len(raw_text_p)
            
            # level_list = [id2pid[max_x], id2pid[max_y]]
            level_list = [max_x, max_y]
            # visit = [start_left,start_right]
            # visited = defaultdict(int)
            # visited[max_x] = 1
            # visited[max_y] = 1
            for level in range(1, no_pad_L - 4 + 1):
                # if level >= 10:
                #     break
                
                logits_f = calculate_level(model, level_list, new_text1[:,:no_pad_L], mask1[:,:no_pad_L]) # no_pad_L, C
                delta_logits = probs[:,label_id] - logits_f[:,label_id] # no_pad_L,1
                delta_logits = delta_logits.squeeze().cpu().numpy().tolist() # no_pad_L
                delta_logits = [args.alpha3 * x + args.alpha1 * y + args.alpha2 * z for x, y, z in zip(delta_logits, sim[:no_pad_L], depth[:no_pad_L])] # no_pad_L
                delta_logits = delta_logits[1:-1] # no_pad_L - 2, respond to raw_text
                delta_p = [(x, y, z + 1) for x, y, z in zip(delta_logits, raw_text, range(len(raw_text)))]
                delta_p = sorted(delta_p, key = lambda z : z[0], reverse = True)
                delta_p = [p for p in delta_p if p[2] not in level_list]
                # delta_tok = [(d, t) for d,t in zip(delta_logits, range(len(new_text1_l)))]
                # filter 1 : remove padding token
                # delta_tok = [p for p in delta_tok if new_text1_l[p[1]] != 101 and new_text1_l[p[1]] != 102]
                # # filter 2 : remove visited token
                # delta_tok = [p for p in delta_tok if p[1] not in level_list]
                # delta_tok = sorted(delta_tok, key = lambda x : x[0], reverse = True)
                # # while len(delta_tok) > 0:
                # #     _ = delta_tok[1]
                # #     if _ not in visit:
                
                tok_p = delta_p[0][1]
                print(f"test_id:{test_id} Level:{level} node:{tok_p}")
                level_list.append(delta_p[0][2])
#             final_tok = [t for t in range(len(new_text1_l)) if t not in level_list and id2tok[new_text1_l[t]] in raw_text]
            end_time = time.time()
            print(f"test_id:{test_id} run_time:{end_time-start_time} seconds")
            avg_time.append(end_time-start_time)
#             final_tok = id2tok[new_text1_l[final_tok[0]]]
#             print(f"test_id:{test_id} Level:{L + 1} node:{final_tok}")

            test_id += 1
    print("avg_time:{:.3f}".format(np.mean(avg_time)))
    # for a1 in alpha1:
    #     for a2 in alpha2:
    #         for a3 in alpha3:
    #             key = str(a1) + "_" + str(a2) + "_" + str(a3)
    #             aopc_ = np.mean(aopcs[key])
    #             print("alpha1:{:d} alpha2:{:d} alpha3:{:d} avg AOPC:{:.3f}".format(a1, a2, a3, aopc_))

