import torch

def tensor_splitter(yb,N,K,train=True):
    trn_idx = []
    for i in range(N):
        inds = (yb==i).nonzero().squeeze()
        if train:
            trains = torch.randperm(inds.shape[0])[:K]
        else:
            trains = torch.arange(inds.shape[0])[:K]
        trn_idx.extend(inds[trains].tolist())
    return trn_idx
