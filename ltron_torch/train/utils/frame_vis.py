import numpy
from datetime import datetime
from pathlib import Path
now = datetime.now()

current_time = now.strftime("%H:%M:%S")

path = Path("~/.cache/ltron/collections/omr_clean/frames/10001-1 - Metroliner@11_2_27_1_7.npz").expanduser()
rollout = numpy.load(path, allow_pickle=True)['rollout'].item()
workspace = rollout['workspace_color_render']
class_ids = self.id_mapping(rollout['workspace_mask_render'], rollout['config']['class'])
save_image(workspace, "imste/" + current_time + ".png")
save_image(class_ids, "imste/" + current_time + ".png")