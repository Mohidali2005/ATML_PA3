import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low rank correction

    When protect_rank is above zero the correction is projected away from the
    leading singular directions of the frozen weight
    """

    def __init__(self,base,rank,alpha,dropout,protect_rank=0):
        super().__init__()
        self.base = base
        self.scale = alpha/rank
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        self.A = nn.Parameter(torch.empty(rank,base.in_features,device=device))
        self.B = nn.Parameter(torch.zeros(base.out_features,rank,device=device))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        if protect_rank > 0:
            U,S,Vh = torch.linalg.svd(base.weight.float(),full_matrices=False)
            k = min(protect_rank,len(S))
            self.register_buffer("Uk",U[:,:k].contiguous())
            self.register_buffer("Vk",Vh[:k].T.contiguous())
        else:
            self.register_buffer("Uk",None)
            self.register_buffer("Vk",None)

    def factors(self):
        """Return the effective factors after the optional projection"""
        A,B = self.A,self.B
        if self.Uk is not None:
            B = B-self.Uk@(self.Uk.T@B)
            A = A-(A@self.Vk)@self.Vk.T
        return A,B

    def forward(self,x):
        # keeping the projection in fp32 so the constraint holds exactly
        with torch.autocast(x.device.type,enabled=False):
            A,B = self.factors()
        update = F.linear(F.linear(self.dropout(x).to(A.dtype),A),B)
        return self.base(x)+self.scale*update.to(x.dtype)


def add_lora(model,targets,rank,alpha,dropout,protect_rank,seed):
    """Freeze the model and wrap every target linear layer with a LoRALinear"""
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    for layer in model.model.layers:
        for parent in (layer.self_attn,layer.mlp):
            for name in targets:
                if hasattr(parent,name):
                    setattr(parent,name,LoRALinear(getattr(parent,name),rank,alpha,dropout,protect_rank))


def lora_modules(model):
    """List the name and module of every LoRALinear"""
    return [(n,m) for n,m in model.named_modules() if isinstance(m,LoRALinear)]


def export_factors(model):
    """Collect the effective factors of every adapter on the cpu"""
    state = {}
    for name,module in lora_modules(model):
        with torch.no_grad():
            A,B = module.factors()
        state[name] = (A.detach().cpu(),B.detach().cpu())
    return state


def merge_factors(model,state,scale):
    """Add the saved corrections into the matching weights of a plain model"""
    modules = dict(model.named_modules())
    for name,(A,B) in state.items():
        weight = modules[name].weight
        delta = scale*(B.float()@A.float())
        weight.data = (weight.data.float()+delta.to(weight.device)).to(weight.dtype)
