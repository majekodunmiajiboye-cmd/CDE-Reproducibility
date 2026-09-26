#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, urllib.request

UPSTREAMS = {
    "adele": ("https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation.git", "a93722320b5c46a371ef94c60c4333def37bbd57"),
    "vericontest": ("https://github.com/HIPREL-Group/VeriContest.git", "b9657edf33b68dbd4992b501967730b30883a04e"),
    "harnessbench": ("https://github.com/Qihoo360/harness-bench.git", "1025086a446653702b80cfb48babbeec35db6b2c"),
}

def clone(name, root):
    url, commit = UPSTREAMS[name]
    dest = root/name
    if not dest.exists(): subprocess.run(["git","clone",url,str(dest)],check=True)
    subprocess.run(["git","-C",str(dest),"fetch","--all","--tags"],check=True)
    subprocess.run(["git","-C",str(dest),"checkout",commit],check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path("inputs/upstream")); args=ap.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    for name in ["adele","vericontest","harnessbench"]: clone(name,args.root)
    hf = args.root/"vericontest_hf_manifest.jsonl"
    if not hf.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/Gax-c/VeriContest/resolve/main/manifest.jsonl", hf)
    print(f"Inputs fetched under {args.root}")
if __name__=="__main__": main()
