import argparse
import csv
import textwrap
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.utils import load_yaml,repo_path,save_json
from task2_vlm.utils import fetch_image,load_suites

PER_SHEET = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--round",type=int,default=0,help="changes the sample after wrong labels were fixed")
    ap.add_argument("--summarize",action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    folder = repo_path(cfg["paths"]["results_dir"])/"label_check"
    folder.mkdir(parents=True,exist_ok=True)
    table = folder/f"label_check_{args.round}.csv"

    if args.summarize:
        with table.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        marks = [r["label_ok"].strip().lower() for r in rows]
        assert all(m in ("ok","wrong") for m in marks),"every row needs ok or wrong in label_ok"
        wrong = [r for r in rows if r["label_ok"].strip().lower() == "wrong"]
        save_json(folder/f"label_check_{args.round}_summary.json",{"checked":len(rows),"ok":marks.count("ok"),"wrong":len(wrong),
                  "wrong_items":[{k:r[k] for k in ("check_id","pair_id","image","question","answer")} for r in wrong]})
        print(f"checked {len(rows)} images and {len(wrong)} were marked wrong")
        return

    suites = load_suites(cfg,["grounding_dev","grounding_test"])
    entries = [(p,it) for s in suites.values() for p in s for it in p["items"]]
    rng = np.random.default_rng([cfg["seed"],args.round])
    chosen = [entries[i] for i in sorted(rng.choice(len(entries),cfg["data"]["label_check_images"],replace=False))]
    with ThreadPoolExecutor(8) as pool:
        images = list(pool.map(lambda e:fetch_image(e[1]["image"]),chosen))

    with table.open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["check_id","pair_id","image","kind","category","question","answer","label_ok"])
        for n,(p,it) in enumerate(chosen):
            writer.writerow([n,p["pair_id"],it["image"],p["kind"],p["category"],p["question"],it["answer"],""])
    for s in range(0,len(chosen),PER_SHEET):
        fig,axes = plt.subplots(2,5,figsize=(22,10))
        for ax,n in zip(axes.flat,range(s,s+PER_SHEET)):
            p,it = chosen[n]
            ax.imshow(images[n])
            for x,y,w,h in it["boxes"]:
                ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="firebrick",linewidth=2))
            ax.set_title(f"#{n}\n"+textwrap.fill(p["question"],34)+f"\nanswer: {it['answer']}",fontsize=13)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(folder/f"sheet_{args.round}_{s//PER_SHEET}.png",dpi=90)
        plt.close(fig)
    print(f"wrote {len(chosen)} images to check in {folder}")


if __name__ == "__main__":
    main()
