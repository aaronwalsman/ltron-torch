import torch.nn as nn

class LtronDetectorConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    DPTConfig,
):
    backbone = 'transformer'
    embedding_dropout = 0.1
    
    dpt_scale_factor = 4
    dpt_blocks = (3,7,11)
    include_dpt_upsample_head = False

class LtronDetector(nn.Module):
    def __init__(self, config, observation_space, checkpoint=None):
        super().__init__()
        
        # save config and observation space
        self.config = config
        self.observation_space = observation_space
        
        # build the embedding
        self.image_embedding = AutoEmbedding(
            config, observation_space['image'])
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # pre decoder layernorm
        self.predecoder_norm = nn.LayerNorm(config.channels)
        
        # detection decoder
        self.decoder = DPT(config)
