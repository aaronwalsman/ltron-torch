#!/usr/bin/env python
import os

import torch
from torch.utils.data import Dataset, DataLoader

import numpy

from PIL import Image

import tqdm

from ltron.hierarchy import len_hierarchy
from ltron.dataset.paths import get_dataset_paths
from ltron.visualization.drawing import block_upscale_image

from ltron_torch.models.positional_encoding import raw_positional_encoding

class CachedDataset(Dataset):
    def __init__(self, data, dataset_name, split, subset=None):
        paths = get_dataset_paths(dataset_name, split, subset=subset)
        self.frame_order = []
        for i in range(len_hierarchy(paths)):
            for j in range(7):
                self.frame_order.append(paths['color_x'][i].replace(
                    '_0000.png', '_%04i.png'%j))
        
        self.data = data
    
    def __len__(self):
        return len(self.frame_order)
    
    def __getitem__(self, index):
        x_path = self.frame_order[index]
        y_path = x_path.replace(
            'color_x_', 'color_y_')[:-9] + '.png'
        x = self.data['frames'][x_path]
        q = self.data['frames'][y_path]
        
        action_path = x_path.replace(
            'color_x_', 'action_').replace('.png', '.npy')
        action_data = numpy.load(action_path, allow_pickle=True).item()
        
        if action_data['pick_and_place']['pick'][0]:
            mode = 1
        elif action_data['vector_offset']['pick'][0]:
            if action_data['vector_offset']['direction'][1] == 1:
                mode = 2
            else:
                mode = 3
        else:
            mode = 0
        
        action = torch.zeros(7, dtype=torch.long)
        pick_location = self.data['snaps'][x_path].get(
            action_data['pick_and_place']['pick'], (0,0))
        place_location = self.data['snaps'][x_path].get(
            action_data['pick_and_place']['place'], (0,0))
        rotate_pick_location = self.data['snaps'][x_path].get(
            action_data['vector_offset']['pick'], (0,0))
        action[0] = pick_location[0]
        action[1] = pick_location[1]
        action[2] = place_location[0]
        action[3] = place_location[1]
        action[4] = rotate_pick_location[0]
        action[5] = rotate_pick_location[1]
        action[6] = mode
        
        return x, q, action, x_path

class Model(torch.nn.Module):
    def __init__(
        self,
        channels=256,
        transformer_1_layers=4,
        transformer_2_layers=6,
        transformer_heads=4,
        transformer_dropout=0.5,
        embedding_dropout=0.1,
    ):
        super(Model, self).__init__()
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        
        self.register_buffer(
            'sequence_encoding',
            raw_positional_encoding(channels, 32**2*2+4).unsqueeze(1),
        )
        
        self.embedding = torch.nn.Embedding(4192, channels)
        self.out_embedding = torch.nn.Embedding(4, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout,
        )
        self.transformer_1 = torch.nn.TransformerEncoder(
            transformer_layer, transformer_1_layers)
        self.transformer_2 = torch.nn.TransformerEncoder(
            transformer_layer, transformer_2_layers)
        
        self.mode_linear = torch.nn.Linear(channels, 4)
        self.pick_place_pick_linear = torch.nn.Linear(channels, 64+64)
        self.pick_place_place_linear = torch.nn.Linear(channels, 64+64)
        self.rotate_pick_linear = torch.nn.Linear(channels, 64+64)
    
    def forward(self, x, q):
        b,h,w = x.shape
        x = x.view(b, h*w).permute(1,0)
        x = self.embedding(x)
        
        q = q.view(b, h*w).permute(1,0)
        q = self.embedding(q)
        
        out = torch.arange(4).view(4,1).expand(4,b).to(x.device)
        out = self.out_embedding(out)
        out_q_x = torch.cat((out, q, x), dim=0)
        out_q_x = out_q_x + self.sequence_encoding
        
        out_q_x = self.transformer_1(out_q_x)
        out = out_q_x[:4]
        out = self.transformer_2(out)
        
        mode = self.mode_linear(out[0])
        pick = self.pick_place_pick_linear(out[1]).view(b, 64, 2)
        place = self.pick_place_place_linear(out[2]).view(b, 64, 2)
        rotate_pick = self.rotate_pick_linear(out[3]).view(b, 64, 2)
        
        return mode, pick, place, rotate_pick

def train(
    num_epochs=50,
):
    print('loading dataset')
    cached_data = numpy.load(
        './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
    train_dataset = CachedDataset(
        cached_data,
        'conditional_snap_two_frames',
        'train',
        subset=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        cached_data,
        'conditional_snap_two_frames',
        'test',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model = Model().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y, paths in iterate:
            x = x.cuda()
            q = q.cuda()
            y = y.cuda()
            
            mode, pick, place, rotate_pick = model(x, q)
            
            total_loss = 0.
            
            mode_target = y[:,6]
            mode_loss = torch.nn.functional.cross_entropy(mode, mode_target)
            total_loss = total_loss + mode_loss
            
            pick_place_batch_entries = (mode_target == 1).unsqueeze(-1)
            num_pick_place = torch.sum(pick_place_batch_entries)
            if num_pick_place:
                pick_target = y[:,:2]
                pick_loss = torch.nn.functional.cross_entropy(
                    pick, pick_target, reduction='none')
                pick_loss = torch.sum(
                    pick_loss * pick_place_batch_entries) / num_pick_place
                pick_loss = pick_loss * 0.125
                total_loss = total_loss + pick_loss
            
                place_target = y[:,2:4]
                place_loss = torch.nn.functional.cross_entropy(
                    place, place_target, reduction='none')
                place_loss = torch.sum(
                    place_loss * pick_place_batch_entries) / num_pick_place
                place_loss = place_loss * 0.125
                total_loss = total_loss + place_loss * 0.125
            
            rotate_pick_batch_entries = (
                (mode_target == 2) | (mode_target == 3)).unsqueeze(-1)
            num_rotate_pick = torch.sum(rotate_pick_batch_entries)
            if num_rotate_pick:
                rotate_pick_target = y[:,4:6]
                rotate_pick_loss = torch.nn.functional.cross_entropy(
                    rotate_pick, rotate_pick_target, reduction='none')
                rotate_pick_loss = (torch.sum(
                    rotate_pick_loss * rotate_pick_batch_entries) /
                    num_rotate_pick)
                rotate_pick_loss = rotate_pick_loss * 0.125
                total_loss = total_loss + rotate_pick_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            iterate.set_description(
                'm: %.02f p: %.02f l: %.02f r: %.03f'%(
                float(mode_loss),
                float(pick_loss),
                float(place_loss),
                float(rotate_pick_loss)))
        
        '''
        print('test')
        model.eval()
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            for x, y, paths in iterate:
                x = x.cuda()
                y = y.cuda().permute(1,0,2)
                #s, b, _ = y.shape
                #y = y.reshape(x*b, 2)
                
                pred = model(x)
                s, b, c = pred.shape
                pred = pred.view(s, b, 64, 2)
                pred = torch.argmax(pred, dim=2).cpu().numpy()
                
                for i in range(b):
                    p = pred[:,i]
                    pred_drawing = numpy.zeros((64, 64, 1))
                    pred_drawing[p[:,0],p[:,1]] = 1.
                    pred_drawing = block_upscale_image(pred_drawing, 256, 256)
                    
                    path = paths[i]
                    image = numpy.array(Image.open(path))
                    image = (
                        image * (1. - pred_drawing) +
                        numpy.array([[[255,0,0]]]) * pred_drawing)
                    image = image.astype(numpy.uint8)
                    image_path = os.path.join(
                        '.', 'epoch_%i_'%epoch + os.path.basename(path))
                    Image.fromarray(image).save(image_path)
                break
        '''
        torch.save(model.state_dict(), 'transformer_locate_model_%04i.pt'%epoch)

if __name__ == '__main__':
    train()
