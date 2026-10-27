import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoImageProcessor,SiglipVisionModel

from common.utils import load_yaml,repo_path,save_json
from task2_vlm.utils import DEVICE,fetch_image,load_suites

CHUNK = 64


def image_keys(cfg):
    """Every image that any suite needs"""
    keys = set()
    for rows in load_suites(cfg).values():
        for r in rows:
            keys.update([r["image"]] if "image" in r else [it["image"] for it in r["items"]])
    return sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--features-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--fake",action="store_true",help="write random features for quick tests")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    folder = repo_path(args.features_dir or cfg["paths"]["data_dir"]+"/features")
    folder.mkdir(parents=True,exist_ok=True)
    results = args.results_dir or cfg["paths"]["results_dir"]
    keys = image_keys(cfg)
    shape = (len(keys),cfg["num_patches"],cfg["feature_dim"])
    array = np.lib.format.open_memmap(folder/"features.npy",mode="w+",dtype=np.float16,shape=shape)
    save_json(folder/"index.json",{k:i for i,k in enumerate(keys)})

    if args.fake:
        rng = np.random.default_rng(cfg["seed"])
        for i in range(0,len(keys),256):
            array[i:i+256] = rng.normal(size=array[i:i+256].shape).astype(np.float16)
        array.flush()
        save_json(f"{results}/tables/feature_cache.json",{"images":len(keys),"total_seconds":0.0,"download_seconds":0.0,"encode_seconds":0.0,
                  "peak_memory_gb":None,"encoder_params":0,"feature_bytes":(folder/"features.npy").stat().st_size,"shape":list(shape)})
        print(f"wrote random features for {len(keys)} images")
        return

    processor = AutoImageProcessor.from_pretrained(cfg["encoder"])
    encoder = SiglipVisionModel.from_pretrained(cfg["encoder"],torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    download_seconds = encode_seconds = 0.0
    start = time.perf_counter()
    with ThreadPoolExecutor(16) as pool:
        for i in range(0,len(keys),CHUNK):
            t = time.perf_counter()
            images = list(pool.map(fetch_image,keys[i:i+CHUNK]))
            download_seconds += time.perf_counter()-t
            t = time.perf_counter()
            pixels = processor(images=images,return_tensors="pt")["pixel_values"].to(DEVICE,encoder.dtype)
            with torch.no_grad():
                array[i:i+len(images)] = encoder(pixel_values=pixels).last_hidden_state.half().cpu().numpy()
            encode_seconds += time.perf_counter()-t
            if (i//CHUNK)%20 == 0:
                print(f"cached {i+len(images)} of {len(keys)} images")
    array.flush()
    save_json(f"{results}/tables/feature_cache.json",{
        "images":len(keys),"total_seconds":time.perf_counter()-start,"download_seconds":download_seconds,"encode_seconds":encode_seconds,
        "peak_memory_gb":torch.cuda.max_memory_allocated()/1e9 if DEVICE == "cuda" else None,
        "encoder_params":sum(p.numel() for p in encoder.parameters()),"feature_bytes":(folder/"features.npy").stat().st_size,
        "shape":list(shape)})
    print(f"cached {len(keys)} images in {time.perf_counter()-start:.0f} seconds")


if __name__ == "__main__":
    main()
