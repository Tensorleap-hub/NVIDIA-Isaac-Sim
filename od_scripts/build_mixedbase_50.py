import json, os, random, shutil, sys

SRC="/home/ubuntu/warehouse3cls_mixedbase"
REAL="/home/ubuntu/warehouse3cls_real"
DST="/home/ubuntu/warehouse3cls_mixedbase_50"
FRAC=0.5
SEED=42

def link_realpath(src_file, dst_file):
    # resolve through existing symlink so dst points at the true asset
    real = os.path.realpath(src_file)
    if os.path.lexists(dst_file):
        os.remove(dst_file)
    os.symlink(real, dst_file)

def build_train():
    ann=json.load(open(f"{SRC}/train/_annotations.coco.json"))
    imgs=ann["images"]
    random.seed(SEED)
    keep=sorted(random.sample(imgs, int(round(len(imgs)*FRAC))), key=lambda i:i["id"])
    keep_ids={i["id"] for i in keep}
    anns=[a for a in ann["annotations"] if a["image_id"] in keep_ids]
    out={"images":keep,"annotations":anns,"categories":ann["categories"]}
    os.makedirs(f"{DST}/train",exist_ok=True)
    json.dump(out, open(f"{DST}/train/_annotations.coco.json","w"))
    for im in keep:
        link_realpath(f"{SRC}/train/{im['file_name']}", f"{DST}/train/{im['file_name']}")
    print(f"train: kept {len(keep)}/{len(imgs)} images, {len(anns)}/{len(ann['annotations'])} anns")
    # composition
    jpg=sum(1 for i in keep if i["file_name"].endswith(".jpg"))
    print(f"  real(jpg)={jpg}  synth(png)={len(keep)-jpg}")

def build_valid():
    # use REAL valid (==traj_v3 valid, 858 imgs) for apples-to-apples eval
    ann=json.load(open(f"{REAL}/valid/_annotations.coco.json"))
    os.makedirs(f"{DST}/valid",exist_ok=True)
    json.dump(ann, open(f"{DST}/valid/_annotations.coco.json","w"))
    for im in ann["images"]:
        link_realpath(f"{REAL}/valid/{im['file_name']}", f"{DST}/valid/{im['file_name']}")
    print(f"valid: {len(ann['images'])} real images (matches traj_v3 eval set)")

if __name__=="__main__":
    build_train(); build_valid()
    print("done ->", DST)
