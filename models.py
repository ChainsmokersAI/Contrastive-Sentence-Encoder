import torch
import torch.nn as nn
import torch.distributed as dist

class SupervisedSimCSE(nn.Module):
    """
    Supervised SimCSE
    """
    def __init__(self, pretrained):
        super().__init__()
        
        # Pre-Trained LM
        self.pretrained=pretrained
        
        # Cosine Similarity
        self.cos_sim=nn.CosineSimilarity(dim=-1)
        # Temperature (Hyperparam)
        self.temp=0.05
        
        # Contrastive Loss
        self.loss=nn.CrossEntropyLoss()
        
    def pooler(self, x):
        # [CLS] without MLP (Hyperparam)
        return x.last_hidden_state[:,0,:]
    
    def get_embedding(self, x):
        # Return Sentence Representation
        x=self.pretrained(x)
        return self.pooler(x)
    
    def forward(self, sent, pos, neg):
        # Forward
        sent=self.pretrained(sent)
        pos=self.pretrained(pos)
        neg=self.pretrained(neg)
        
        # Pooling
        # Shape: batch_size x hidden_dim
        repr_sent=self.pooler(sent)
        repr_pos=self.pooler(pos)
        repr_neg=self.pooler(neg)

        # Multi-GPU
        if dist.is_initialized():
            repr_list_sent=[torch.zeros_like(repr_sent) for _ in range(dist.get_world_size())]
            repr_list_pos=[torch.zeros_like(repr_pos) for _ in range(dist.get_world_size())]
            repr_list_neg=[torch.zeros_like(repr_neg) for _ in range(dist.get_world_size())]

            # All Gather
            dist.all_gather(tensor_list=repr_list_sent, tensor=repr_sent.contiguous())
            dist.all_gather(tensor_list=repr_list_pos, tensor=repr_pos.contiguous())
            dist.all_gather(tensor_list=repr_list_neg, tensor=repr_neg.contiguous())

            # Grad Fn
            repr_list_sent[dist.get_rank()]=repr_sent
            repr_list_pos[dist.get_rank()]=repr_pos
            repr_list_neg[dist.get_rank()]=repr_neg
            
            # Shape: (world_size * batch_size) x hidden_dim
            repr_sent=torch.cat(repr_list_sent, dim=0)
            repr_pos=torch.cat(repr_list_pos, dim=0)
            repr_neg=torch.cat(repr_list_neg, dim=0)

        # Cosine Similarity
        sim_pos=self.cos_sim(repr_sent.unsqueeze(1), repr_pos.unsqueeze(0))/self.temp
        sim_neg=self.cos_sim(repr_sent.unsqueeze(1), repr_neg.unsqueeze(0))/self.temp
        
        # Contrastive Loss
        sim=torch.cat([sim_pos, sim_neg], dim=1)
        label=torch.arange(sim.size(0)).long().to(dist.get_rank())
        loss=self.loss(sim, label)
        
        return loss
