from torch.nn import Module, Linear, Embedding, Dropout

class TileEmbedding(Module):
    def __init__(self, tile_h, tile_w, tile_c, channels, dropout):
        super().__init__()
        self.tile_linear = Linear(tile_h * tile_w * tile_c, channels)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        s, b, h, w, c = x.shape
        x = x.view(s, b, h*w*c)
        x = self.tile_linear(x)
        x = self.dropout(x)
        return x

class TokenEmbedding(Module):
    def __init__(self, vocabulary, channels, dropout):
        super().__init__()
        self.embedding = Embedding(vocabulary, channels)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x
