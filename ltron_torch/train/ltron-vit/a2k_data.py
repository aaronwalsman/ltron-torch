import numpy as np
import cv2
import glob


path = 'data/'

images = glob.glob(path+"*.png")
images.sort()
masks = [n.replace(".png", ".npy").replace("color", "label") for n in images]

sequences = {}

for img, mask in zip(images, masks):
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

for seq in sequences.values():
    np.save(seq["file_name"], seq)




