import torch

from ltron_torch.models.mlp import LinearStack, Conv2dStack
from ltron_torch.models.spatial import SE3Layer
from ltron_torch.models.simple_fcn import SimpleFCN
from ltron_torch.models.seq_fcn import SeqFCN, SeqGlobal
from ltron_torch.models.vit_transformer import VITTransformer
from ltron_torch.models.step_model import StepModel
from ltron_torch.models.seq_model import SeqModel

def backbone_feature_dim(name):
    if 'resnet18' in name:
        return 512
    elif 'resnet34' in name:
        return 512
    elif 'resnet50' in name:
        return 2048
    elif 'simple_fcn' in name:
        return backbone_feature_dim('resnet50')
    elif 'vit' in name:
        return 256

def seq_model(
    backbone_name,
    num_classes,
    input_width=256,
    input_height=256,
    transformer_channels=256,
    decoder_channels=256,
    viewpoint_head=False,
    pose_head=False,
    pretrained=True,
):
    # build the heads
    dense_heads = {
        'x' : torch.nn.Identity(),
        'class_label' : Conv2dStack(
            3, decoder_channels, decoder_channels, num_classes),
        'confidence' : Conv2dStack(
            3, decoder_channels, decoder_channels, 1),
    }
    single_heads = {}
    global_heads = {}
    if viewpoint_head:
        dim = backbone_feature_dim(backbone_name)
        single_heads['viewpoint'] = torch.nn.Linear(dim, 7)
    if pose_head:
        dense_heads['pose'] = torch.nn.Sequential(
            Conv2dStack(3, decoder_channels, decoder_channels, 9),
            SE3Layer(dim=1),
        )
        global_heads['global_pose'] = torch.nn.Sequential(
            LinearStack(3, transformer_channels, transformer_channels, 9),
            SE3Layer(dim=1),
        )
    
    # build the backbone
    if backbone_name == 'transformer_fcn':
        backbone = SeqFCN(
            transformer_channels=transformer_channels,
            decoder_channels=decoder_channels,
            pretrained=pretrained,
        )
    elif backbone_name == 'simple_fcn':
        backbone = SeqGlobal(
            decoder_channels=decoder_channels,
            pretrained=pretrained,
        )
    
    #elif backbone_name == 'vit':
    #    backbone = VITTransformer(
    #        channels=decoder_channels,
    #        compute_single=compute_single,
    #    )
    else:
        raise NotImplementedError
    
    # return
    return SeqModel(
        backbone=backbone,
        add_spatial_embedding=True,
        decoder_channels=decoder_channels,
        dense_heads=dense_heads,
        single_heads=single_heads,
        global_heads=global_heads,
    )

def single_step_model(
    backbone_name,
    num_classes,
    input_width=256,
    input_height=256,
    decoder_channels=256,
    viewpoint_head=False,
    pose_head=False,
    pretrained=True,
):
    
    # build the heads
    dense_heads = {
        'x' : torch.nn.Identity(),
        'class_label' : Conv2dStack(
            3, decoder_channels, decoder_channels, num_classes),
        'confidence' : Conv2dStack(
            3, decoder_channels, decoder_channels, 1, detach_input=True),
    }
    single_heads = {}
    if viewpoint_head:
        dim = backbone_feature_dim(backbone_name)
        single_heads['viewpoint'] = torch.nn.Linear(dim, 7)
    if pose_head:
        dense_heads['pose'] = torch.nn.Sequential(
            Conv2dStack(3, decoder_channels, decoder_channels, 9),
            SE3Layer(dim=1),
        )
    
    # build the backbone
    compute_single = (len(single_heads) != 0)
    if backbone_name == 'simple_fcn':
        backbone = SimpleFCN(
            decoder_channels=decoder_channels,
            pretrained=pretrained,
            compute_single=compute_single
        )
    
    elif backbone_name == 'vit':
        backbone = VITTransformer(
            channels=decoder_channels,
            compute_single=compute_single,
        )
    else:
        raise NotImplementedError
    
    # return
    return StepModel(
        backbone=backbone,
        add_spatial_embedding=True,
        decoder_channels=decoder_channels,
        dense_heads=dense_heads,
        single_heads=single_heads,
    )

def camera_test_step_model(
    backbone_name,
    num_classes,
    input_width=256,
    input_height=256,
    decoder_channels=256,
    pretrained=True,
):
    
    dense_heads = {}
    dim = backbone_feature_dim(backbone_name)
    single_heads = {
        'viewpoint' : torch.nn.Linear(dim, 7),
        'class_label' : torch.nn.Linear(dim, num_classes),
    }
    
    if backbone_name == 'simple_fcn':
        backbone = SimpleFCN(
            decoder_channels=decoder_channels,
            pretrained=pretrained,
            compute_single=True
        )
    
    elif backbone_name == 'vit':
        backbone = VITTransformer(
            channels=decoder_channels,
            compute_single=compute_single,
        )
    
    else:
        raise NotImplementedError
    
    return StepModel(
        backbone=backbone,
        add_spatial_embedding=True,
        decoder_channels=decoder_channels,
        dense_heads=dense_heads,
        single_heads=single_heads,
    )
