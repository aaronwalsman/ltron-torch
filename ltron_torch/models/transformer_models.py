import torch

from ltron_torch.models.positional_encoding import positional_encoding
from ltron_torch.models.mlp import LinearStack

class SparseImageSequenceEncoder(torch.nn.Module):
    def __init__(
        self,
        tokens_per_image,
        max_seq_length,
        n_read_tokens,
        read_channels,
        read_from_input=False,
        channels=512,
        residual_channels=2048,
        vocabulary_size=4096,
        num_layers=12,
        num_heads=12,
        embedding_dropout=0.1,
        transformer_dropout=0.5,
    ):
        super(SparseImageSequenceEncoder, self).__init__()
        self.tokens_per_image = tokens_per_image
        self.num_read_tokens = num_read_tokens
        self.channels = channels
        self.residual_channels = residual_channels
        self.read_from_input = read_from_input
        
        self.register_buffer(
            'positional_encoding',
            positional_encoding(
                channels, max_seq_length, tokens_per_image).unsqueeze(2)
        )
        
        self.token_embedding = torch.nn.Embedding(vocabulary_size, channels)
        self.read_embedding = torch.nn.Embedding(num_read_tokens, channels)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            num_heads,
            residual_channels,
            transformer_dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer,
            num_layers,
        )
        
        # this stack is probably harmful... should be residual blocks?
        self.read_head = LinearStack(3, channels, channels, read_channels)
    
    def forward(self, x, mask=None):
        s, hw, b = x.shape
        assert hw == self.tokens_per_image
        
        x = x.view(s*hw, b)
        x = self.token_embedding(x) * self.channels**0.5
        x = x + self.positional_encoding[:s].view(s*hw, 1, self.channels)
        
        r = self.read_embedding.weight.unsqueeze(1) * self.channels**0.5
        r = r.expand(self.num_read_tokens, b, self.channels)
        
        rx = torch.cat((r,x), dim=0)
        rx = self.transformer(rx, mask=mask)
        
        if self.read_from_input:
            r = rx
        else:
            r = rx[:self.num_read_tokens]
        r = self.read_head(r)
        
        return r

class ImageSequenceEncoder(torch.nn.Module):
    
    def __init__(
        self,
        tokens_per_image,
        max_seq_length,
        n_read_tokens,
        read_channels,
        read_from_input=False,
        channels=512,
        residual_channels=2048,
        vocabulary_size=4096,
        num_layers=6,
        num_heads=4,
        embedding_dropout=0.1,
        transformer_dropout=0.5,
    ):
        super(ImageSequenceEncoder, self).__init__()
        self.tokens_per_image = tokens_per_image
        self.num_read_tokens = num_read_tokens
        self.channels = channels
        self.residual_channels = residual_channels
        self.read_from_input = read_from_input
        
        self.register_buffer(
            'positional_encoding',
            positional_encoding(
                channels, max_seq_length, tokens_per_image).unsqueeze(2)
        )
        
        self.token_embedding = torch.nn.Embedding(vocabulary_size, channels)
        self.read_embedding = torch.nn.Embedding(num_read_tokens, channels)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            num_heads,
            residual_channels,
            transformer_dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer,
            num_layers,
        )
        
        #self.read_linear = torch.nn.Linear(channels, read_channels)
        self.read_head = LinearStack(4, channels, channels, read_channels)
    
    def forward(self, x, mask=None):
        s, hw, b = x.shape
        assert hw == self.tokens_per_image
        
        x = x.view(s*hw, b)
        x = self.token_embedding(x) * self.channels**0.5
        x = x + self.positional_encoding[:s].view(s*hw, 1, self.channels)
        
        r = self.read_embedding.weight.unsqueeze(1) * self.channels**0.5
        r = r.expand(self.num_read_tokens, b, self.channels)
        
        rx = torch.cat((r,x), dim=0)
        rx = self.transformer(rx, mask=mask)
        
        if self.read_from_input:
            r = rx
        else:
            r = rx[:self.num_read_tokens]
        r = self.read_head(r)
        
        return r
