#!/usr/bin/env python
import os
from collections import OrderedDict

import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import numpy

import PIL.Image as Image

import splendor.masks as masks

from ltron.visualization.drawing import block_upscale_image, draw_crosshairs
from ltron.hierarchy import len_hierarchy
from ltron.dataset.paths import get_dataset_paths

#from ltron_torch.models.transformer_models import ImageSequenceEncoder
from ltron_torch.models.transformer import (
    TokenMapSequenceEncoder,
    TransformerConfig,
)
from ltron_torch.models.transformer_masks import neighborhood
import ltron_torch.models.dvae as dvae

def extract_path_indices(path, num_indices):
    parts = path.split('_')[-num_indices:]
    parts = tuple([int(part.split('.')[0]) for part in parts])
    return parts

def shorten_cache_data():

    print('loading dataset')
    cached_data = numpy.load(
        './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
    
    print('generating shorter paths')
    paths = cached_data['frames'].keys()
    short_paths = []
    for path in tqdm.tqdm(paths):
        if 'color_x' in path:
            i, j = extract_path_indices(path, 2)
        elif 'color_y' in path:
            i, = extract_path_indices(path, 1)
        if i < 1000:
            short_paths.append(path)
    
    new_data = {}
    new_data['frames'] = {
        path:cached_data['frames'][path] for path in short_paths}
    new_data['snaps'] = {
        path:cached_data['snaps'][path] for path in short_paths}
    
    numpy.savez_compressed('./dvae_small_cache.npz', new_data)

class CachedDataset(Dataset):
    def __init__(self, dataset_name, split):
        
        # ----------------------------------------------------------------------
        print('loading dataset')
        cached_data = numpy.load(
            './dvae_small_cache.npz', allow_pickle=True)['arr_0'].item()
        subset=1000
        # ----------------------------------------------------------------------
        #print('loading dataset')
        #cached_data = numpy.load(
        #    './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
        #subset=None
        # ----------------------------------------------------------------------
    
        paths = get_dataset_paths(dataset_name, split, subset=subset)
        self.frame_order = []
        for i in range(len_hierarchy(paths)):
            for j in range(7):
                self.frame_order.append(paths['color_x'][i].replace(
                    '_0000.png', '_%04i.png'%j))
        
        self.data = cached_data
    
    def __len__(self):
        return len(self.frame_order)
    
    def __getitem__(self, index):
        x_path = self.frame_order[index]
        y_path = x_path.replace(
            'color_x_', 'color_y_')[:-9] + '.png'
        x = self.data['frames'][x_path]
        q = self.data['frames'][y_path]
        
        snaps = torch.zeros((8,3), dtype=torch.long)
        
        for i, snap in enumerate((
            (1,5),
            (1,6),
            (1,7),
            (1,8),
            (2,1),
            (2,2),
            (3,1),
            (3,2),
        )):
            if snap in self.data['snaps'][x_path]:
                yy, xx = self.data['snaps'][x_path][snap]
                snaps[i] = torch.LongTensor([yy,xx,1])
            else:
                snaps[i] = torch.LongTensor([0,0,0])
        
        return x, q, snaps, x_path


def visualize():
    print('making dataset')
    dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    
    for i, (x, q, snaps, path) in enumerate(dataset):
        image = Image.open(path)
        
        blocks = masks.color_index_to_byte(x)
        blocks = block_upscale_image(blocks, 256, 256)
        
        image = numpy.concatenate((image, blocks), axis=1)
        out_path = os.path.basename(path).replace('color_x_', 'vis_')
        Image.fromarray(image).save(out_path)
        
        if i > 16:
            break

# Dense Pick ===================================================================

class DensePickModel(torch.nn.Module):
    def __init__(self, channels=1024):
        super(DensePickModel, self).__init__()
        #self.decoder = ImageSequenceEncoder(
        #    tokens_per_image=32*32,
        #    max_seq_length=1,
        #    n_read_tokens=0,
        #    read_channels=9,
        #    read_from_input=True,
        #    channels=channels,
        #    num_layers=12,
        #)
        config = TransformerConfig(
            decoder_channels = 9,
            num_layers=6,
            channels=512,
            num_heads=8,
        )
        self.decoder = TokenMapSequenceEncoder(config)
    
    def forward(self, x):
        return self.decoder(x)

class DenseConvModel(torch.nn.Module):
    def __init__(self):
        super(DenseConvModel, self).__init__()
        self.embedding = torch.nn.Embedding(4096, 128)
        num_groups = 4
        blocks_per_group = 2
        num_layers = num_groups * blocks_per_group
        groups = OrderedDict()
        groups['group_1'] = dvae.DecoderGroup(
            128, 256*8, blocks_per_group, num_layers, upsample=False)
        groups['group_2'] = dvae.DecoderGroup(
            256*8, 256*4, blocks_per_group, num_layers, upsample=False)
        groups['group_3'] = dvae.DecoderGroup(
            256*4, 256*2, blocks_per_group, num_layers, upsample=False)
        groups['group_4'] = dvae.DecoderGroup(
            256*2, 256, blocks_per_group, num_layers, upsample=False)
        groups['out'] = torch.nn.Sequential(OrderedDict([
            ('relu', torch.nn.ReLU()),
            ('conv', torch.nn.Conv2d(256, 9, kernel_size=1)),
        ]))
        
        self.groups = torch.nn.Sequential(groups)
    
    def forward(self, x):
        x = self.embedding(x)
        s, hw, b, c = x.shape
        x = x.view(hw, b, c).permute(1,2,0).reshape(b,c,32,32)
        x = self.groups(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(2,0,1).contiguous()
        
        return x

def train_dense_pick(
    num_epochs=100,
):
    print('making dataset')
    train_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model = DensePickModel().cuda()
    #model = DenseConvModel().cuda()
    #model.load_state_dict(torch.load('./dense_02/model_0050.pt'))
    
    #attention_mask = neighborhood(32, 32, width=3).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    running_snap_loss = 0.
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, snaps, paths in iterate:
            x = x.cuda()
            
            b, h, w = x.shape
            x = x.view(b, h*w).permute(1,0).unsqueeze(0)
            
            snaps = snaps.permute(1, 0, 2).contiguous().cuda()
            #snap_locations = (snaps[:,:,:2] - 31.5) / 31.5
            snap_locations = snaps[:,:,:2]
            snap_valid = snaps[:,:,[2]]#.view(8*b)
            
            #x = model(x, mask=attention_mask)
            x = model(x)
            
            total_loss = 0.
            
            snap_target = torch.zeros(32*32, b, dtype=torch.long).cuda()
            for i in range(8):
                yx = snap_locations[i]
                yy = (yx[:,0] / 2.).long()
                xx = (yx[:,1] / 2.).long()
                yx = yy * 32 + xx
                snap_target[yx, range(b)] = i + 1
            
            class_weight = torch.ones(9)
            class_weight[0] = 0.1
            
            snap_loss = torch.nn.functional.cross_entropy(
                x.view(32*32*b, -1),
                snap_target.view(32*32*b),
                weight=class_weight.cuda()
            )
            
            total_loss = total_loss + snap_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_snap_loss = running_snap_loss * 0.9 + float(snap_loss) * 0.1
            iterate.set_description('s: %.04f'%float(running_snap_loss))
        
        torch.save(model.state_dict(), 'model_%04i.pt'%epoch)
        
        with torch.no_grad():
            model.eval()
            iterate = tqdm.tqdm(test_loader)
            for x, q, snaps, paths in iterate:
                x = x.cuda()
                
                b, h, w = x.shape
                x = x.view(b, h*w).permute(1,0).unsqueeze(0)
                
                x = model(x)
                
                #x = torch.softmax(x, dim=-1)
                #x = x.view(8, b, 64, 64)
                #snap_maps = torch.sum(x, dim=0).unsqueeze(-1).cpu().numpy()
                
                snaps = snaps.permute(1, 0, 2).contiguous().cuda()
                snap_locations = snaps[:,:,:2]
                
                for i in range(b):
                    path = paths[i]
                    image = numpy.array(Image.open(path))
                    
                    #x_map = torch.softmax(x[:,i], dim=-1).cpu().numpy()
                    
                    x_class = torch.argmax(x[:,i], dim=-1).cpu().numpy()
                    x_class = x_class.reshape(32, 32)
                    x_background = x_class == 0
                    class_color = masks.color_index_to_byte(x_class)
                    class_color = block_upscale_image(class_color, 256, 256)
                    x_background = block_upscale_image(x_background, 256, 256)
                    x_background = x_background.reshape(256, 256, 1)
                    image = (
                        image * x_background +
                        class_color * (1. - x_background)
                    ).astype(numpy.uint8)
                    
                    #p_normal = x_map[:,0].reshape(32, 32, 1)
                    #p_normal = block_upscale_image(p_normal, 256, 256)
                    #image = image * p_normal + (1. - p_normal) * [0,0,255]
                    image = image.astype(numpy.uint8)
                    
                    '''
                    #snap_map = snap_maps[i]
                    #snap_map = block_upscale_image(snap_map, 256, 256)
                    #image = image * (1. - snap_map) + [0,0,255] * snap_map
                    
                    neg_path = path.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz')
                    neg_snap_map = numpy.load(neg_path, allow_pickle=True)
                    neg_snap_map_i = neg_snap_map['arr_0'][:,:,0]
                    #neg_snap_map_s = neg_snap_map['arr_0'][:,:,1]
                    neg_snap_map = (neg_snap_map_i == 2) | (neg_snap_map_i == 3)
                    
                    pos_path = path.replace(
                        'color_x_', 'snap_pos_').replace(
                        '.png', '.npz')
                    pos_snap_map = numpy.load(pos_path, allow_pickle=True)
                    pos_snap_map_i = pos_snap_map['arr_0'][:,:,0]
                    #pos_snap_map_s = pos_snap_map['arr_0'][:,:,1]
                    pos_snap_map = pos_snap_map_i == 1
                    
                    gt_dense_snap = neg_snap_map | pos_snap_map
                    gt_dense_snap = gt_dense_snap.reshape(64, 64, 1)
                    gt_dense_snap = block_upscale_image(
                        gt_dense_snap, 256, 256)
                    gt_dense_snap = gt_dense_snap * 0.5
                    image = (
                        image * (1. - gt_dense_snap) +
                        [0,255,0] * gt_dense_snap
                    )
                    
                    gt_snaps = numpy.zeros(4096)
                    gt_snaps[snap_locations[:,i].cpu()] = 1
                    gt_snaps = gt_snaps.reshape(64, 64, 1)
                    gt_snaps = block_upscale_image(gt_snaps, 256, 256)
                    gt_snaps = gt_snaps * 0.5
                    image = image * (1. - gt_snaps) + [255,0,255] * gt_snaps
                    '''
                    image = image.astype(numpy.uint8)
                    Image.fromarray(image).save(
                        './tmp_%04i_%04i.png'%(epoch, i))
                
                break

# Sparse Pick ==================================================================

class SparsePickModel(torch.nn.Module):
    def __init__(self, channels=512):
        super(SparsePickModel, self).__init__()
        self.encoder = ImageSequenceEncoder(
            max_seq_length=1,
            tokens_per_image=32*32,
            num_read_tokens=8,
            read_channels=2,
            read_from_input=False,
            channels=channels,
        )
    
    def forward(self, x):
        return self.encoder(x)

def train_sparse_pick(
    num_epochs=100,
):
    print('making dataset')
    train_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model = SparsePickModel().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, snaps, paths in iterate:
            x = x.cuda()
            
            b, h, w = x.shape
            x = x.view(b, h*w).permute(1,0).unsqueeze(0)
            
            snaps = snaps.permute(1, 0, 2).contiguous().cuda()
            #snap_locations = snaps[:,:,0] * 64 + snaps[:,:,1]
            snap_locations = (snaps[:,:,:2] - 31.5) / 31.5
            snap_valid = snaps[:,:,[2]]#.view(8*b)
            snap_targets = snap_locations
            #snap_targets = torch.FloatTensor([
            #    [-1.0,-1.0],
            #    [-0.8,-0.8],
            #    [-0.6,-0.6],
            #    [-0.4,-0.4],
            #    [-0.2,-0.2],
            #    [ 0.0, 0.0],
            #    [ 0.2, 0.2],
            #    [ 0.4, 0.4]]).unsqueeze(1).expand(8,b,2).cuda()
            
            x = model(x)
            
            total_loss = 0.
            
            #snap_loss = torch.nn.functional.mse_loss(
            #    x, snap_targets, reduction='none')
            #snap_loss = torch.nn.functional.mse_loss(x, snap_targets)
            snap_loss = torch.nn.functional.smooth_l1_loss(x, snap_targets)
            #snap_loss = torch.nn.functional.cross_entropy(
            #    x.view(8*b,4096), snap_locations.view(8*b), reduction='none')
            
            #if epoch == 2:
            #    import pdb
            #    pdb.set_trace()
            
            #snap_loss = torch.sum(snap_loss * snap_valid)/torch.sum(snap_valid)
            total_loss = total_loss + snap_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            iterate.set_description('s: %.04f'%float(snap_loss))
        
        torch.save(model.state_dict(), 'model_%04i.pt'%epoch)
        
        with torch.no_grad():
            model.eval()
            iterate = tqdm.tqdm(test_loader)
            for x, q, snaps, paths in iterate:
                x = x.cuda()
                
                b, h, w = x.shape
                x = x.view(b, h*w).permute(1,0).unsqueeze(0)
                
                x = model(x)
                
                #x = torch.softmax(x, dim=-1)
                #x = x.view(8, b, 64, 64)
                #snap_maps = torch.sum(x, dim=0).unsqueeze(-1).cpu().numpy()
                
                snaps = snaps.permute(1, 0, 2).contiguous().cuda()
                snap_locations = snaps[:,:,0] * 64 + snaps[:,:,1]
                
                for i in range(b):
                    path = paths[i]
                    image = numpy.array(Image.open(path))
                    
                    #snap_map = snap_maps[i]
                    #snap_map = block_upscale_image(snap_map, 256, 256)
                    #image = image * (1. - snap_map) + [0,0,255] * snap_map
                    
                    neg_path = path.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz')
                    neg_snap_map = numpy.load(neg_path, allow_pickle=True)
                    neg_snap_map_i = neg_snap_map['arr_0'][:,:,0]
                    #neg_snap_map_s = neg_snap_map['arr_0'][:,:,1]
                    neg_snap_map = (neg_snap_map_i == 2) | (neg_snap_map_i == 3)
                    
                    pos_path = path.replace(
                        'color_x_', 'snap_pos_').replace(
                        '.png', '.npz')
                    pos_snap_map = numpy.load(pos_path, allow_pickle=True)
                    pos_snap_map_i = pos_snap_map['arr_0'][:,:,0]
                    #pos_snap_map_s = pos_snap_map['arr_0'][:,:,1]
                    pos_snap_map = pos_snap_map_i == 1
                    
                    gt_dense_snap = neg_snap_map | pos_snap_map
                    gt_dense_snap = gt_dense_snap.reshape(64, 64, 1)
                    gt_dense_snap = block_upscale_image(
                        gt_dense_snap, 256, 256)
                    gt_dense_snap = gt_dense_snap * 0.5
                    image = (
                        image * (1. - gt_dense_snap) +
                        [0,255,0] * gt_dense_snap
                    )
                    
                    gt_snaps = numpy.zeros(4096)
                    gt_snaps[snap_locations[:,i].cpu()] = 1
                    gt_snaps = gt_snaps.reshape(64, 64, 1)
                    gt_snaps = block_upscale_image(gt_snaps, 256, 256)
                    gt_snaps = gt_snaps * 0.5
                    image = image * (1. - gt_snaps) + [255,0,255] * gt_snaps
                    
                    for j in range(8):
                        yy, xx = x[j, i].cpu().numpy()
                        print(xx, yy)
                        yy = yy * 127.5 + 127.5
                        xx = xx * 127.5 + 127.5
                        draw_crosshairs(image, xx, yy, 5, [0, 0, 255])
                    print('------')
                    
                    image = image.astype(numpy.uint8)
                    Image.fromarray(image).save(
                        './tmp_%04i_%04i.png'%(epoch, i))
                
                break

if __name__ == '__main__':
    #shorten_cache_data()
    #visualize()
    train_dense_pick()
    #train_sparse_pick()
