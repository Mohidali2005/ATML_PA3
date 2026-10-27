import torch
import torch.nn.functional as F
from torch import nn

from task1_finetune.lora import add_lora
from task2_vlm.utils import DEVICE,IMAGE_PAD,USE_AMP,collate_prompts,load_llm


class Connector(nn.Sequential):
    """Two layer perceptron that maps encoder patch features to language model embeddings"""

    def __init__(self,in_dim,out_dim):
        super().__init__(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Linear(out_dim,out_dim))


class VLM(nn.Module):
    """A frozen language model that reads visual tokens produced by a trainable connector"""

    def __init__(self,llm,connector,tokenizer):
        super().__init__()
        self.llm = llm
        self.connector = connector
        self.tokenizer = tokenizer
        self.image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def visual_tokens(self,features):
        """Connector output for a stack of patch features"""
        weight = self.connector[0].weight
        return self.connector(features.to(weight.device,weight.dtype))

    def embeds(self,input_ids,features):
        """Text embeddings with the image placeholders replaced by visual tokens"""
        emb = self.llm.get_input_embeddings()(input_ids)
        if features is None:
            return emb
        vis = self.visual_tokens(features)
        self.visual_norm = vis.detach().float().norm(dim=-1).mean()
        return emb.masked_scatter((input_ids == self.image_id).unsqueeze(-1),vis.reshape(-1,vis.size(-1)).to(emb.dtype))

    def example_nll(self,batch,features,include_end=True):
        """Mean token loss over the scored tokens of every example"""
        batch = {k:v.to(self.llm.device) for k,v in batch.items()}
        hidden = self.llm.model(inputs_embeds=self.embeds(batch["input_ids"],features),attention_mask=batch["attention_mask"]).last_hidden_state[:,:-1]
        targets = batch["labels"][:,1:]
        keep = targets != -100
        if not include_end:
            keep &= targets != self.end_id
        token_loss = F.cross_entropy(self.llm.lm_head(hidden[keep]).float(),targets[keep],reduction="none")
        rows = torch.arange(len(targets),device=targets.device).unsqueeze(1).expand_as(targets)[keep]
        total = torch.zeros(len(targets),device=targets.device).index_add(0,rows,token_loss)
        return total/keep.sum(1).clamp(min=1)

    @torch.no_grad()
    def generate(self,prompts,features,max_new_tokens):
        """Greedy answers for a batch of prompts"""
        ids,mask = collate_prompts(prompts,self.tokenizer.pad_token_id)
        ids,mask = ids.to(self.llm.device),mask.to(self.llm.device)
        out = self.llm.generate(inputs_embeds=self.embeds(ids,features),attention_mask=mask,max_new_tokens=max_new_tokens,
                                do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=self.tokenizer.pad_token_id)
        return [t.strip() for t in self.tokenizer.batch_decode(out,skip_special_tokens=True)]


@torch.no_grad()
def calibrate_connector(vlm,features,token_ids):
    """Rescale the last connector layer so visual tokens match the size of text embeddings

    The rule uses training data only. The mean norm of the visual tokens is set to the mean norm of the
    embeddings of the training caption tokens. Returns the measured norms and the applied scale
    """
    text_norm = vlm.llm.get_input_embeddings().weight[token_ids].float().norm(dim=-1).mean().item()
    visual_norm = vlm.visual_tokens(features).float().norm(dim=-1).mean().item()
    scale = text_norm/visual_norm
    last = vlm.connector[2]
    last.weight.mul_(scale)
    last.bias.mul_(scale)
    return {"text_norm":text_norm,"visual_norm_before":visual_norm,"scale":scale,"visual_norm_after":vlm.visual_tokens(features).float().norm(dim=-1).mean().item()}


def add_adapters(vlm,cfg):
    """Add the trainable low rank adapters to the attention projections of the language model"""
    lora = cfg["lora"]
    add_lora(vlm.llm,lora["targets"],lora["rank"],lora["alpha"],lora["dropout"],0,cfg["seed"])


def build_vlm(cfg,tokenizer,tiny=False,lora=False):
    """Load the frozen language model and a new connector

    The language model stays in half precision on the gpu and the connector and adapters stay in full precision
    """
    llm = load_llm(cfg,torch.float16 if USE_AMP else torch.float32,tiny).to(DEVICE)
    for p in llm.parameters():
        p.requires_grad = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    llm.config.use_cache = False
    torch.manual_seed(cfg["seed"])
    vlm = VLM(llm,Connector(cfg["feature_dim"],llm.config.hidden_size).to(DEVICE),tokenizer)
    if lora:
        add_adapters(vlm,cfg)
    return vlm


def trainable_state(vlm):
    """Copy of every trainable parameter on the cpu"""
    return {n:p.detach().cpu().clone() for n,p in vlm.named_parameters() if p.requires_grad}


def load_trainable(vlm,state):
    """Copy saved values into the matching parameters"""
    for n,p in vlm.named_parameters():
        if n in state:
            p.data.copy_(state[n])
