import io
from collections import Counter

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from common.utils import load_json,read_jsonl,repo_path
from task1_finetune.utils import prompt_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
SESSION = requests.Session()
SESSION.mount("http://",HTTPAdapter(max_retries=Retry(total=6,backoff_factor=1,status_forcelist=[429,500,502,503,504]),pool_maxsize=32))
SUITES = ["captions_train","captions_dev","presence_pool","counting_pool","vqa_train","grounding_dev","grounding_test","vqa_dev","vqa_test"]


def amp():
    """Half precision autocast on the gpu and nothing on the cpu"""
    return torch.autocast(DEVICE,dtype=torch.float16,enabled=USE_AMP)


def load_tokenizer(cfg):
    """Load the tokenizer of the language model"""
    return AutoTokenizer.from_pretrained(cfg["llm"])


def load_llm(cfg,dtype,tiny=False):
    """Load the language model or a random tiny model with the same layer layout for quick tests"""
    if tiny:
        config = AutoConfig.from_pretrained(cfg["llm"])
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 8
        config.num_hidden_layers = 28
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(config).to(dtype)
    return AutoModelForCausalLM.from_pretrained(cfg["llm"],torch_dtype=dtype)


def load_suites(cfg,names=SUITES):
    """Read the selected data of every suite"""
    return {n:read_jsonl(cfg["paths"]["data_dir"]+f"/{n}.jsonl") for n in names}


def image_url(key):
    """Web address of the image behind a feature store key"""
    split,image_id = key.rsplit("_",1)
    return f"http://images.cocodataset.org/{split}/COCO_{split}_{int(image_id):012d}.jpg"


def fetch_image(key):
    """Download one image as an rgb picture and retry when the connection drops"""
    response = SESSION.get(image_url(key),timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


class FeatureStore:
    """Cached encoder features of every image kept in one memory mapped array"""

    def __init__(self,folder):
        folder = repo_path(folder)
        self.index = load_json(folder/"index.json")
        self.array = np.load(folder/"features.npy",mmap_mode="r")

    def get(self,keys):
        """Stack the features of a list of images into one fp16 tensor"""
        return torch.from_numpy(np.stack([self.array[self.index[k]] for k in keys]))


def question_text(cfg,example):
    """The text that follows the visual tokens"""
    if example["kind"] == "caption":
        return cfg["prompts"]["caption"]
    return example["question"]+"\n"+cfg["prompts"]["answer_suffix"]


def encode_prompt(tokenizer,cfg,example,with_image=True,num_patches=None):
    """Token ids of the user turn with the visual tokens in front of the question"""
    text = question_text(cfg,example)
    if with_image:
        text = VISION_START+IMAGE_PAD+VISION_END+"\n"+text
    ids = tokenizer(prompt_text(tokenizer,text),add_special_tokens=False)["input_ids"]
    if with_image:
        pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        at = ids.index(pad)
        ids = ids[:at]+[pad]*(num_patches or cfg["num_patches"])+ids[at+1:]
    return ids


def encode_example(tokenizer,cfg,example,with_image=True,num_patches=None):
    """Token ids and labels where only the answer and the end of turn token are scored"""
    prompt = encode_prompt(tokenizer,cfg,example,with_image,num_patches)
    answer = tokenizer(example["answer"]+"<|im_end|>",add_special_tokens=False)["input_ids"]
    return {"input_ids":prompt+answer,"labels":[-100]*len(prompt)+answer}


def collate_prompts(prompts,pad_id):
    """Pad prompts on the left so that generation starts at the same position"""
    longest = max(len(p) for p in prompts)
    ids = [[pad_id]*(longest-len(p))+p for p in prompts]
    mask = [[0]*(longest-len(p))+[1]*len(p) for p in prompts]
    return torch.tensor(ids),torch.tensor(mask)


def caption_examples(rows):
    """Turn caption rows into examples"""
    return [{"kind":"caption","image":r["image"],"answer":r["caption"]} for r in rows]


def pair_examples(pair):
    """Turn one pair into its two examples"""
    return [{"kind":pair["kind"],"image":it["image"],"question":pair["question"],"answer":it["answer"]} for it in pair["items"]]


def sample_update(cfg,stage,index,update,suites,captions):
    """Examples of one update

    The draws depend only on the stage and the update number so a resumed run sees the same data
    """
    size = cfg["microbatch_size"]*cfg["accum_steps"]
    if stage["name"] == "caption":
        start = update*size
        picks = [np.random.default_rng([cfg["seed"],index,(start+j)//len(captions)]).permutation(len(captions))[(start+j)%len(captions)] for j in range(size)]
        return [captions[int(i)] for i in picks],Counter({"caption":size//2})
    rng = np.random.default_rng([cfg["seed"],index,update])
    names = list(stage["mixture"])
    chosen = rng.choice(names,size=size//2,p=[stage["mixture"][n] for n in names])
    pools = {"presence":suites["presence_pool"],"counting":suites["counting_pool"],"vqa":suites["vqa_train"]}
    examples = []
    for name in chosen:
        if name == "caption":
            examples += [captions[int(i)] for i in rng.integers(len(captions),size=2)]
        else:
            examples += pair_examples(pools[name][int(rng.integers(len(pools[name])))])
    return examples,Counter(chosen.tolist())
