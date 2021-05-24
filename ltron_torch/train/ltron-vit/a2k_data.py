import numpy as np
import cv2
import glob
import tqdm
import sys


path = '../../envs/fork/' if len(sys.argv) == 1 else sys.argv[1]

images = glob.glob(path+"*.png")
images.sort()
masks = [n.replace(".png", ".npy").replace("color", "label") for n in images]

print(len(images), len(masks))

sequences = {}

for img, mask in tqdm.tqdm(zip(images, masks), total=len(images)):
    key = int(img.split("_")[1])
    seq_pos = int(img.split("_")[2].split(".")[0])
    img_data = np.moveaxis(cv2.imread(img), 2, 0) / 255.0
    category_data = np.load(mask).reshape(-1)
    position_data = np.stack([[i, seq_pos] for i in range(32*32)])
    patches = np.stack([img_data[:, (i%32)*8:((i%32)+1)*8, (i//32)*8:((i//32)+1)*8] for i in range(32*32)])

    if key not in sequences:
        sequences[key] = {
            "file_name": mask.replace("label", "data"),
            "images": patches,
            "categories": category_data,
            "image_positions": position_data,
            "last_categories": category_data.reshape(-1),
        }
    else:
        difference_mask = sequences[key]["last_categories"] != category_data
        sequences[key]["last_categories"] = category_data
        sequences[key]["images"] = np.concatenate((sequences[key]["images"], patches[difference_mask]))
        sequences[key]["categories"] = np.concatenate((sequences[key]["categories"], category_data[difference_mask]))
        sequences[key]["image_positions"] = np.concatenate((sequences[key]["image_positions"], position_data[difference_mask]))

max_len = max([seq["images"].shape[0] for seq in sequences.values()])

for seq in sequences.values():
    images = np.zeros((max_len, *seq["images"].shape[1:]), dtype=np.float32)
    images[:seq["images"].shape[0]] = np.array(seq["images"], dtype=np.float32)
    categories = np.zeros((max_len, *seq["categories"].shape[1:]), dtype=np.long)
    categories[:seq["categories"].shape[0]] = np.array(seq["categories"], dtype=np.long)
    image_positions = np.zeros((max_len, *seq["image_positions"].shape[1:]), dtype=np.long)
    image_positions[:seq["image_positions"].shape[0]] = np.array(seq["image_positions"], dtype=np.long)

    np.save(seq["file_name"].replace("data_", "data-images_"), images)
    np.save(seq["file_name"].replace("data_", "data-categories_"), categories)
    np.save(seq["file_name"].replace("data_", "data-image-positions_"), image_positions)





