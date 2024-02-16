import torch


def get_mst(alpha1,alpha2,alpha3,sim,depth,logits,batch_text):
    if logits is not None:
        sim = [alpha1 * x1 + alpha2 * x2 + alpha3 * x3 for x1,x2,x3 in zip(sim,depth,logits)]
    else:
        sim = [alpha1 * x1 + alpha2 * x2 for x1,x2 in zip(sim,depth)]
    mst = [(tok,score) for tok,score in zip(batch_text,sim)]
    if alpha3 > 0:
        mst = sorted(mst, key = lambda x : x[1], reverse=True)
    else:
        mst = sorted(mst, key = lambda x : x[1])
    mst = [tok for tok,_ in mst if tok != "[PAD]" and tok != "[CLS]" and tok != "[SEP]"]
    return mst

class EdgeNode:
    def __init__(self,x,y,v) -> None:
        self.x = x
        self.y = y
        self.v = v

    def __lt__(self, other):
        return other.v > self.v

@torch.no_grad()
def calculate_level(model, level_list, input, mask):
    # input : 1, L, d
    # mask : 1, L, L
    # pred : func
    # level_list : List[int]
    n = input.size(1)
    identity_matrix = torch.eye(n).reshape(1, n, n)
    mask_ = 1 - identity_matrix
    for level in level_list:
        mask_[:, :, level] = 0
    mask = mask * mask_
    variant_tensor = input * mask # L,L,d
    mask = mask.expand(n, n,-1) # L,L,L
    logits = model(variant_tensor, mask) # 
    return logits # L,C