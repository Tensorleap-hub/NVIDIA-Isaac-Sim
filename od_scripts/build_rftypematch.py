"""Build a synth-TYPE control set: full real (4110) + random-frame synth
count-matched to traj_v3's synth (1056). Only difference vs traj_v3 is
random-frame vs trajectory synth. valid = pure real valid (858)."""
import json, os, random

SRC="/home/ubuntu/warehouse3cls_mixedbase"      # source of real(jpg)+random-frame synth(png)
REAL="/home/ubuntu/warehouse3cls_real"
DST="/home/ubuntu/warehouse3cls_rftypematch"
N_SYNTH=1056                                     # == traj_v3 synth frame count
SEED=42

def link_realpath(src_file, dst_file):
    real=os.path.realpath(src_file)
    if os.path.lexists(dst_file): os.remove(dst_file)
    os.symlink(real, dst_file)

def build_train():
    ann=json.load(open(f"{SRC}/train/_annotations.coco.json"))
    real=[i for i in ann["images"] if i["file_name"].endswith(".jpg")]
    synth=[i for i in ann["images"] if not i["file_name"].endswith(".jpg")]
    random.seed(SEED)
    synth_keep=random.sample(synth, N_SYNTH)
    keep=sorted(real+synth_keep, key=lambda i:i["id"])
    keep_ids={i["id"] for i in keep}
    anns=[a for a in ann["annotations"] if a["image_id"] in keep_ids]
    os.makedirs(f"{DST}/train", exist_ok=True)
    json.dump({"images":keep,"annotations":anns,"categories":ann["categories"]},
              open(f"{DST}/train/_annotations.coco.json","w"))
    for im in keep:
        link_realpath(f"{SRC}/train/{im['file_name']}", f"{DST}/train/{im['file_name']}")
    print(f"train: {len(real)} real + {len(synth_keep)} random-frame synth = {len(keep)} imgs, {len(anns)} anns")

def build_valid():
    ann=json.load(open(f"{REAL}/valid/_annotations.coco.json"))
    os.makedirs(f"{DST}/valid", exist_ok=True)
    json.dump(ann, open(f"{DST}/valid/_annotations.coco.json","w"))
    for im in ann["images"]:
        link_realpath(f"{REAL}/valid/{im['file_name']}", f"{DST}/valid/{im['file_name']}")
    print(f"valid: {len(ann['images'])} real images (== traj_v3 eval set)")

if __name__=="__main__":
    build_train(); build_valid(); print("done ->", DST)
