from torch.nn import Linear, LayerNorm, Embedding
from torch.optim import AdamW

from ltron_torch.models.parameter import NoWeightDecayParameter

class OptimizerConfig:
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

def adamw_optimizer(model, config):
    # mostly from minGPT
    decay_names = set()
    no_decay_names = set()
    decay_params = []
    no_decay_params = []
    decay_modules = (Linear,)
    no_decay_modules = (LayerNorm, Embedding)
    for module_name, module in model.named_modules():
        is_decay_module = isinstance(module, decay_modules)
        is_no_decay_module = isinstance(module, no_decay_modules)
        for param_name, param in module.named_parameters():
            full_param_name = (
                '%s.%s'%(module_name, param_name)
                if module_name else param_name
            )

            if isinstance(param, NoWeightDecayParameter):
                no_decay_names.add(full_param_name)
                no_decay_params.append(param)
            elif param_name.endswith('bias'):
                no_decay_names.add(full_param_name)
                no_decay_params.append(param)
            elif param_name.endswith('weight') and is_decay_module:
                decay_names.add(full_param_name)
                decay_params.append(param)
            elif param_name.endswith('weight') and is_no_decay_module:
                no_decay_names.add(full_param_name)
                no_decay_params.append(param)

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
    optimizer = AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=config.betas,
    )
    return optimizer
