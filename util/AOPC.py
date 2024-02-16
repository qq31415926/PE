
from copy import deepcopy
import torch

def calculate_metric(args,predictor,label_id,probs,tree,aopc_token,pad_token,s_text,metric="AOPC",s_text_a=None,s_text_b=None):
    
    aopc_delta = []
    top = args.top
    if metric == "AOPC": 
        pool_size = args.pool_size
        k = max(int(len(tree) * top * pool_size), 1)
        # k = len(tree)
        pool = []
        # aopc_delta_per_example = []
        s_text_s = deepcopy(s_text)
        if s_text_a is not None:
            # print("s_text_a:{} ")
            s_text_a_ = deepcopy(s_text_a)
        if s_text_b is not None:
            s_text_b_ = deepcopy(s_text_b)
            
        print("#"*15)
        # if s_text_a is not None and s_text_b is not None:
        #     print("s_text_a:{} s_text_b:{}".format(s_text_a, s_text_b))
        for j in range(k):
            
            tok = tree[j]
            if aopc_token == "del":

                if tok in s_text_s:
                    s_text_s.remove(tok)
            
                mask = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s]
                # mask_ms = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_ms]
                if s_text_a is not None:
                    if tok in s_text_a_:
                        s_text_a_.remove(tok)
                    elif tok in s_text_b_:
                        s_text_b_.remove(tok)
            else:
                for s_text_tok in range(len(s_text_s)):
                    
                    if s_text_s[s_text_tok] == tok:
                        s_text_s[s_text_tok] = pad_token
                mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s ]

                if s_text_a is not None and s_text_b is not None:
                    f = True
                    for tid in range(len(s_text_a_)):
                        if s_text_a_[tid] == tok:
                            s_text_a_[tid] = pad_token
                            f = False
                            break
                    if f:
                        for tid in range(len(s_text_b_)):
                            if s_text_b_[tid] == tok:
                                s_text_b_[tid] = pad_token
                                break
                
                        # break
                        # cnt_tok1 -= 1
                
            
            with torch.no_grad():
                if args.dataset == "ptb" or args.dataset == "trec" or args.dataset == "yelp":
                    probs_i = predictor(input=s_text_s,mask = mask) 
                elif args.dataset == "snli":
                    probs_i = predictor(input = s_text_a_, mask = [1 if x_ != "[PAD]" else 0 for x_ in s_text_a_], input_else = s_text_b_, mask_else = [1 if x_ != "[PAD]" else 0 for x_ in s_text_b_])
                # print("tok:{} ")
                delta_p = probs[0,label_id].item()-probs_i[0,label_id].item()
                # print("tok:{} delta:{:.3f}".format(tok, delta_p))
                pool.append((tok,delta_p))
                # aopc_delta.append(delta_p)
                
            # print("text:{} tree size:{:d} tree edge1:{} tree edge2:{} p1:{:.4f} p2:{:.4f} AOPC:{:.4f}".format(s_text,len(tree),tok1,tok2,probs[0,label_id].item(),probs_i[0,label_id].item(),aopc_delta[-1]))
            # print("text:{}  tok:{} p1:{:.4f} p2:{:.4f} AOPC:{:.4f}".format(s_text,tok,probs[0,label_id].item(),probs_i[0,label_id].item(),aopc_delta[-1]))
    # return aopc_delta1,suff_delta1,logodd_delta1,flag_ms,flag_es
        
        pool = sorted(pool, key= lambda x : x[1], reverse=True)
        candidate_size = max(int(len(pool) / pool_size), 1)
        # for i,candidate_tok, score in enumerate(pool[:candidate_size]):
        #     print("dist_tok:{} tok:{} AOPC:{:.4f}".format(mst[i],candidate_tok,score))
        aopc_delta = [score for _,score in pool][:candidate_size]
        return aopc_delta
    else:
        pass