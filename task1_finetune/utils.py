import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

CONDITIONS = ["baseline","full_mid","full","l2sp","lora4","lora16","oplora"]


def load_tokenizer(cfg):
    """Load the tokenizer of the base model"""
    return AutoTokenizer.from_pretrained(cfg["model"])


def load_model(cfg,dtype,tiny=False,path=None):
    """Load the base model or a random tiny model with the same layer layout for quick tests"""
    if tiny:
        config = AutoConfig.from_pretrained(cfg["model"])
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 8
        config.num_hidden_layers = 28
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(config).to(dtype)
    return AutoModelForCausalLM.from_pretrained(path or cfg["model"],torch_dtype=dtype)


def prompt_text(tokenizer,query,system=None):
    """Apply the chat template with thinking turned off and the assistant turn left open"""
    messages = [{"role":"user","content":query}]
    if system:
        messages = [{"role":"system","content":system}]+messages
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)


def encode_example(tokenizer,query,response):
    """Return token ids and labels where only the response and the end of turn token are scored"""
    prompt_ids = tokenizer(prompt_text(tokenizer,query),add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response+"<|im_end|>",add_special_tokens=False)["input_ids"]
    return {"input_ids":prompt_ids+response_ids,"labels":[-100]*len(prompt_ids)+response_ids}


def collate(examples,pad_id):
    """Pad a list of encoded examples on the right into one batch"""
    longest = max(len(e["input_ids"]) for e in examples)
    ids,mask,labels = [],[],[]
    for e in examples:
        gap = longest-len(e["input_ids"])
        ids.append(e["input_ids"]+[pad_id]*gap)
        mask.append([1]*len(e["input_ids"])+[0]*gap)
        labels.append(e["labels"]+[-100]*gap)
    return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(mask),"labels":torch.tensor(labels)}


def model_path(cfg,work_dir,condition):
    """Folder or hub name of the fp16 model for one evaluation condition"""
    if condition == "baseline":
        return cfg["model"]
    if condition == "full_mid":
        return f"{work_dir}/full/mid_model"
    if condition in ("full","l2sp"):
        return f"{work_dir}/{condition}/final_model"
    return f"{work_dir}/{condition}/merged_model"
