import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Any, Union, Callable
from model import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MLP, LaneNet
from torch.autograd import Variable

class TransTraj (nn.Module):
    """Transformer with a linear embedding

    Args:
        nn (_type_): _description_
    """
    def __init__(self, pose_dim: int, n_future: int,
                 d_model = 512, nhead = 8, N = 6, dim_feedforward = 2048, dropout=0.1):
        """_summary_

        Args:
            pose_dim (int): Pose dimension [x,y, yaw, v ...] || Encoder input size
            dec_out_size (int): Decoder output size
            num_queries (int): Number of trajectories
            future_size (int, optional): the output trajectory size
            d_model (int, optional): the number of expected features in the input -> embedding dimension. Defaults to 512.
            nhead (int, optional): the number of heads in the multiheadattention models. Defaults to 8.
            N (int, optional): the number of sub-encoder-layers in the encoder -> number of nn.TransformerEncoderLayer in nn.TransformerEncoder. Defaults to 6.
            dim_feedforward (int, optional): dimension of the feedforward network model in nn.TransformerEncoder. Defaults to 2048.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        # ----------------------------------------------------------------------- #
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        self.enc_linear_embedding = LinearEmbedding(pose_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.dec_linear_embedding = LinearEmbedding(pose_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        # Trajectory proposal
        self.query_embed = nn.Embedding(n_future, d_model) 
        
        # Create a Transformer module (in the paper ins only the middle box, not include linear output in the decoder)
        self.transformer = nn.Transformer (d_model=d_model, nhead=nhead, num_encoder_layers=N, num_decoder_layers=N, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation='relu')
        
        # self.linear_out = nn.Linear(d_model, dec_out_size)
        d_model_2 = int (d_model / 2)
        self.linear_out = nn.Linear(d_model, pose_dim, bias=True)
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def process_lanes (self, lanes: torch.Tensor):
        # Get the lanes in vectorized form ( VectorNet: v = [dsi , dei , ai, j] )
        lane_v = torch.cat (([lanes[:, :, :-1, :2], lanes[:, :, 1:, :2], lanes[:, :, 1:, 2:]]), dim=-1)
        # Process the lanes to get the features
        lane_feature: torch.Tensor = self.subgraph(lane_v)
        return lane_feature
        
    def forward(self, historic_traj: torch.Tensor, future_traj: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor, 
                src_padding_mask: Optional[torch.Tensor] = None, tgt_padding_mask: Optional[torch.Tensor] = None):
        """_summary_

        Args:
            historic_traj (torch.Tensor): Historical trajectories [bs, 50, num_features]
            future_traj (torch.Tensor): Future trajectories [bs, 60, num_features]
            lanes (torch.Tensor): Polylines of lanes [bs, max_lanes_num, num_lines, lane_features]
            src_mask (torch.Tensor): _description_
            tgt_mask (torch.Tensor): _description_
            src_padding_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            tgt_padding_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # ----------------------------------------------------------------------- #
        # Apply linear embedding with the positional encoding
        src = self.enc_linear_embedding(historic_traj)
        src = self.pos_encoder(src)
        # ----------------------------------------------------------------------- #
        # Repeat BS times
        self.query_batches = self.query_embed.weight.view(1, *self.query_embed.weight.shape).repeat(future_traj.shape[0], 1, 1)
        self.query_batches = self.pos_decoder(self.query_batches)
        # ----------------------------------------------------------------------- #
        # Transformer step
        output = self.transformer(src, self.query_batches, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, 
                                  tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        # ----------------------------------------------------------------------- #
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
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + Variable(self.pe[:x.size(1)], requires_grad=False)
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

class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))