import shutil
import pathlib
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("create a subset of a folder")
parser.add_argument("src", type=str)
parser.add_argument("size", type=int)
parser.add_argument("dest", type=str)
parser.add_argument("rglob", type=str)
args = parser.parse_args()

path = Path(args.src).expanduser()
Path(args.dest).mkdir(parents=False, exist_ok=True)
counter = 0
for p in path.rglob(args.rglob):
    new_path = args.dest + "/" + str(p).split("/")[-1]
    shutil.move(str(p), new_path)
    counter += 1
    if counter >= args.size:
        break
