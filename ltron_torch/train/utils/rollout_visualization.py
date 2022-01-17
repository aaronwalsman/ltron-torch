from pathlib import Path
import 

target_folder = Path("~/.cache/ltron/collections/omr_clean/frames").expanduser().rglob(".npz")

for f in target_folder:
    
