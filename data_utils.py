import pandas as pd

import torch
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    """
    Dataset for Supervised Contrastive Learning
    """
    def __init__(self, path, tokenizer):
        # Triplet: Sentence, Positive, Hard Negative
        self.sent=[]
        self.pos=[]
        self.neg=[]
        
        # Load Data
        df_sup=pd.read_csv(path)
        
        for index in df_sup.index:
            data=df_sup.loc[index]
            
            self.sent.append(tokenizer.encode(data['sent0']))
            self.pos.append(tokenizer.encode(data['sent1']))
            self.neg.append(tokenizer.encode(data['hard_neg']))
    
    def __getitem__(self, idx):
        return self.sent[idx], self.pos[idx], self.neg[idx]
    
    def __len__(self):
        return len(self.sent)

def collate_fn_supervised(pad_token_id):
    def collate_fn(batch):
        """
        Same Sequence Length on Same Batch (Supervised Setting)
        """
        max_len_sent=0
        max_len_pos=0
        max_len_neg=0
        for sent, pos, neg in batch:
            if len(sent)>max_len_sent: max_len_sent=len(sent)
            if len(pos)>max_len_pos: max_len_pos=len(pos)
            if len(neg)>max_len_neg: max_len_neg=len(neg)
                
        batch_sent=[]
        batch_pos=[]
        batch_neg=[]
        for sent, pos, neg in batch:
            sent.extend([pad_token_id]*(max_len_sent-len(sent)))
            batch_sent.append(sent)
            
            pos.extend([pad_token_id]*(max_len_pos-len(pos)))
            batch_pos.append(pos)
            
            neg.extend([pad_token_id]*(max_len_neg-len(neg)))
            batch_neg.append(neg)
            
        return torch.tensor(batch_sent), torch.tensor(batch_pos), torch.tensor(batch_neg)

    return collate_fn
