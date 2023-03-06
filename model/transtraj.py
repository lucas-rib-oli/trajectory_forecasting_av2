import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Any, Union, Callable

class TransTraj (nn.Module):
    """Transformer with a linear embedding

    Args:
        nn (_type_): _description_
    """
    def __init__(self, enc_inp_size : int, dec_inp_size : int, dec_out_size : int, out_traj_size: int, d_model = 512, nhead = 8, N = 6, dim_feedforward = 2048, dropout=0.1):
        """_summary_

        Args:
            enc_inp_size (int): Encoder input size
            dec_inp_size (int): Decoder input size
            dec_out_size (int): Decoder output size
            d_model (int, optional): the number of expected features in the input -> embedding dimension. Defaults to 512.
            nhead (int, optional): the number of heads in the multiheadattention models. Defaults to 8.
            N (int, optional): the number of sub-encoder-layers in the encoder -> number of nn.TransformerEncoderLayer in nn.TransformerEncoder. Defaults to 6.
            dim_feedforward (int, optional): dimension of the feedforward network model in nn.TransformerEncoder. Defaults to 2048.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        self.enc_linear_embedding = LinearEmbedding(enc_inp_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.dec_linear_embedding = LinearEmbedding(dec_inp_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        # Create a Transformer module (in the paper ins only the middle box, not include linear output in the decoder)
        self.transformer = nn.Transformer (d_model=d_model, nhead=nhead, num_encoder_layers=N, num_decoder_layers=N, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation='gelu')
        
        # self.linear_out = nn.Linear(d_model, dec_out_size)
        d_model_2 = int (d_model / 2)
        self.linear_out = nn.Sequential( nn.Linear(d_model, d_model, bias=True), 
                                         nn.LayerNorm(d_model), 
                                         nn.GELU(), 
                                         nn.Linear(d_model, d_model_2, bias=True), 
                                         nn.Linear(d_model_2, dec_out_size, bias=True) )
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.num_queries = out_traj_size
        self.query_embed = nn.Embedding(self.num_queries, d_model)        
        # self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)
        
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask: Optional[torch.Tensor] = None, tgt_padding_mask: Optional[torch.Tensor] = None):
        # Apply linear embedding with the positional encoding
        src = self.enc_linear_embedding(src)
        src = self.pos_encoder(src)
        # tgt = self.dec_linear_embedding(tgt)
        # tgt = self.pos_decoder(tgt)
        
        self.query_batches = self.query_embed.weight.view(1, *self.query_embed.weight.shape).repeat(tgt.shape[0], 1, 1)
        
        output = self.transformer(src, self.query_batches, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, 
                                  tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        # Get the output i the expected dimensions
        output = self.linear_out(output)

        return output
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.pos_encoder (self.enc_linear_embedding(src)), src_mask)
    
    def decode (self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.pos_decoder(self.dec_linear_embedding(tgt)), memory, tgt_mask)

    def generate (self, out_dec : torch.Tensor):
        return self.linear_out(out_dec)
    
class PositionalEncoding(nn.Module):
    """PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. 
       The positional encodings have the same dimension as the embeddings so that the two can be summed. 
       Here, we use sine and cosine functions of different frequencies.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LinearEmbedding(nn.Module):
    """Linear embedding for the sequence.
    Allow pass to [x, y, z] to [x0, x1, ..., x512] for example 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)

class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit, bias=True),
            nn.LayerNorm(hidden_unit),
            nn.GELU(),
            nn.Linear(hidden_unit, hidden_unit, bias=True)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask