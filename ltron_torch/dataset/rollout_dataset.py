import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy
from PIL import Image
from splendor.image import save_image
import splendor
from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron_torch.config import Config
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform, default_image_untransform)
import pdb

class rolloutFramesConfig(Config):
    train_split = 'rollouts_frames'
    train_subset = None
    start = 0
    batch_size = 4
    epochs = 4
    checkpoint_frequency = 10000000
    test_frequency = 100
    visualization_frequency = 100
    end = None
    file_override = None
    loader_workers = 0
    dataset = "omr_clean"
    test_split = "pico"

class rolloutFrames(Dataset):

    def id_mapping(self, arr, id_map):
        return numpy.vectorize(id_map.__getitem__)(arr)

    def color_mapping(self, arr, id_map):
        return numpy.vectorize(id_map.__getitem__)(arr)

    def __init__(self, dataset, split, subset, transform=None):

        self.transform = transform
        self.workspace = []
        # self.pos_snap = []
        # self.neg_snap = []
        # self.mask = []
        self.stack_label = []
        dataset_paths = get_dataset_paths(dataset, split, subset=subset)
        self.rollout_paths = dataset_paths[split]

    def __len__(self):
        return len(self.rollout_paths)

    def __getitem__(self, i):
        
        path = self.rollout_paths[i]
        rollout = numpy.load(path, allow_pickle=True)['rollout'].item()
        workspace = rollout['workspace_color_render']
        pos_snap_reduced = numpy.where(rollout['workspace_pos_snap_render'][:, :, 0] > 0, 1, 0)
        neg_snap_reduced = numpy.where(rollout['workspace_neg_snap_render'][:, :, 0] > 0, 1, 0)
        class_ids = self.id_mapping(rollout['workspace_mask_render'], rollout['config']['class'])
        color_ids = self.color_mapping(rollout['workspace_mask_render'], rollout['config']['color'])
        stacked_label = numpy.stack([class_ids, pos_snap_reduced, neg_snap_reduced, color_ids], axis=2)


        if self.transform is not None:
            workspace = self.transform(workspace)
            return workspace, stacked_label

        return workspace, stacked_label

def build_rolloutFrames_train_loader(config, batch_overload=None):
    print('-'*80)
    print("Building single frame data loader")
    dataset = rolloutFrames(
            config.dataset,
            config.train_split,
            config.train_subset,
            default_image_transform,
    )

    loader = DataLoader(
            dataset,
            batch_size = batch_overload if batch_overload else config.batch_size,
            num_workers = config.loader_workers,
            shuffle=True,
    )
    
    return loader



def main():
    config = rolloutFramesConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    loader = build_rolloutFrames_train_loader(config, 1)
    counter = 1
    for workspace, label in loader:
        print(workspace.type())
        print(label.type())
        # pdb.set_trace()
        image = numpy.transpose(workspace[0].squeeze().detach().cpu().numpy(), [1,2,0])
        im = numpy.uint8(image * 255)
        im = default_image_untransform(workspace[0])
        save_image(im, "test_dataset/test_im" + str(counter) + ".png")
        mask = label[0, :, :, 2].squeeze().detach().cpu().numpy()
        mask = numpy.uint8(mask * 255)
        save_image(mask, "test_dataset/test_mask" + str(counter) + ".png")
        print(numpy.where(label[0, :, :, 0] > 0))
        print(numpy.where(label[0, :, :, 1] > 0))
        print(workspace.shape)
        print(label.shape)
        print(numpy.unique(label[0, :, :, 0]))
        print(numpy.unique(label[0, :, :, 1]))
        print(numpy.unique(label[0, :, :, 2]))
        counter += 1


if __name__ == '__main__' :
    main()



