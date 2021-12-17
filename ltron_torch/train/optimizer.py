from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Embedding
from torch.optim import AdamW

from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.config import Config

class OptimizerConfig(Config):
    optimizer = 'adamw'
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)

def build_optimizer(model, config):
    
    decay_names = set()
    no_decay_names = set()
    decay_params = []
    no_decay_params = []
    #decay_modules = (Linear, Conv2d)
    no_decay_modules = (
        BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Embedding)
    for module_name, module in model.named_modules():
        #is_decay_module = isinstance(module, decay_modules)
        is_no_decay_module = isinstance(module, no_decay_modules)
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = (
                '%s.%s'%(module_name, param_name)
                if module_name else param_name
            )

            if isinstance(param, NoWeightDecayParameter):
                no_decay_names.add(full_param_name)
                no_decay_params.append(param)
            elif param_name.endswith('bias') or is_no_decay_module:
                no_decay_names.add(full_param_name)
                no_decay_params.append(param)
            else:
                decay_names.add(full_param_name)
                decay_params.append(param)
            #elif param_name.endswith('weight') and is_decay_module:
            #    decay_names.add(full_param_name)
            #    decay_params.append(param)
            #elif param_name.endswith('weight') and is_no_decay_module:
            #    no_decay_names.add(full_param_name)
            #    no_decay_params.append(param)
            #else:
            #    print(module)
            #    print(module_name, param_name)
            #    raise Exception('Something bad happened')

    param_intersection = decay_names & no_decay_names
    param_union = decay_names | no_decay_names
    assert len(param_intersection) == 0
    assert len(param_union) == len(decay_names) + len(no_decay_names)

    optimizer_groups = [
        {'params': decay_params,
         'weight_decay':config.weight_decay,
        },
        {'params': no_decay_params,
         'weight_decay':0.,
        },
    ]
    
    if config.optimizer == 'adam':
        optimizer = Adam(
            optimizer_groups,
            lr=config.learning_rate,
            betas=config.betas,
        )
    elif config.optimizer == 'adamw':
        optimizer = AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=config.betas,
        )
    else:
        raise ValueError('Unexpected optimizer "%s"'%config.optimizer)
    
    return optimizer
