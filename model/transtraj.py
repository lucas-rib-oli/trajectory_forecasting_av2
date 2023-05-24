import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from typing import Optional, Any, Union, Callable
from model import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MLP, LaneNet
from model.axial_attention_module import AxialTransformerEncoder, AxialTransformerEncoderLayer

class TransTraj (nn.Module):
    """Transformer with a linear embedding

    Args:
        nn (_type_): _description_
    """
    def __init__(self, pose_dim: int, dec_out_size: int, num_queries: int,
                 lane_channels: int, subgraph_width: int, num_subgraph_layers: int,
                 future_size: int = 60,
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
        self.future_size = future_size
        self.d_model = d_model
        self.pose_dim = pose_dim
        self.dec_out_size = dec_out_size
        # ----------------------------------------------------------------------- #
        self.enc_linear_embedding = LinearEmbedding(pose_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        # ----------------------------------------------------------------------- #
        # VectorNet Subgraph
        self.subgraph = LaneNet(lane_channels, subgraph_width, num_subgraph_layers)
        self.lanes_emb = LinearEmbedding(subgraph_width*2, d_model)
        lane_enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                 batch_first=True)
        lane_dec_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                 with_self_att=True,
                                                 batch_first=True)
        self.lane_encoder = TransformerEncoder(encoder_layer=lane_enc_layer, num_layers=N)
        self.lane_decoder = TransformerDecoder(decoder_layer=lane_dec_layer, num_layers=N)
        # ----------------------------------------------------------------------- #
        # Use the Axial Tranformer Encoder Layer
        encoder_layer = AxialTransformerEncoderLayer (d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, num_dimensions = 2, embed_dim_index = -1,
                                                      batch_first=True)
        decoder_layer = TransformerDecoderLayer (d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                 with_self_att=True,
                                                 batch_first=True)

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=N)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=N)
        
        
        # self.linear_out = nn.Linear(d_model, dec_out_size)
        d_model_2 = int (d_model / 2)
        # self.linear_out = nn.Sequential( nn.Linear(d_model, d_model, bias=True), 
        #                                  nn.LayerNorm(d_model), 
        #                                  nn.GELU(), 
        #                                  nn.Linear(d_model, d_model_2, bias=True), 
        #                                  nn.Linear(d_model_2, dec_out_size, bias=True) )
        
        self.reg_mlp = nn.Sequential(
                       nn.Linear(d_model, d_model*2, bias=True),
                       nn.LayerNorm(d_model*2),
                       nn.ReLU(),
                       nn.Dropout(dropout),
                       nn.Linear(d_model*2, d_model, bias=True),
                       nn.Linear(d_model, dec_out_size, bias=True), # Dim_features (x,y ..) * N Frames futuros --> view (60, pose_dim (2 or 6))
                       nn.Dropout(dropout)) 
        
        self.cls_FFN = PointerwiseFeedforward(d_model, 2*d_model, dropout=dropout)
        self.classification_layer = nn.Sequential(
                                    nn.Linear(d_model, d_model),
                                    nn.Linear(d_model, 1, bias=True))
        self.cls_opt = nn.Softmax(dim=-1)
        
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.num_queries = num_queries # Number trajectories
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        # self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)
    
    def process_lanes (self, lanes: torch.Tensor):
        # Get the lanes in vectorized form ( VectorNet: v = [dsi , dei , ai, j] )
        lane_v = torch.cat (([lanes[:, :, :-1, :2], lanes[:, :, 1:, :2], lanes[:, :, 1:, 2:]]), dim=-1)
        # Process the lanes to get the features
        lane_feature: torch.Tensor = self.subgraph(lane_v)
        return lane_feature
        
    def forward(self, historic_traj: torch.Tensor, future_traj: torch.Tensor, lanes: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor, 
                src_padding_mask: Optional[torch.Tensor] = None, tgt_padding_mask: Optional[torch.Tensor] = None):
        """_summary_

        Args:
            historic_traj (torch.Tensor): Historical trajectories [BS, A, H, num_features]
            future_traj (torch.Tensor): Future trajectories [BS, A, F, num_features]
            lanes (torch.Tensor): Polylines of lanes [BS, max_lanes_num, num_lines, lane_features]
            src_mask (torch.Tensor): _description_
            tgt_mask (torch.Tensor): _description_
            src_padding_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            tgt_padding_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Get dimensions
        bs = historic_traj.shape[0]
        num_agents = historic_traj.shape[1]
        historic_timesteps = historic_traj.shape[2]
        num_features = historic_traj.shape[3]
        # ----------------------------------------------------------------------- #
        # Apply linear embedding with the positional encoding
        historic_traj = self.enc_linear_embedding(historic_traj) # [BS, A, H, d_model]      
        historic_traj = self.pos_encoder(historic_traj)
        # tgt = self.dec_linear_embedding(tgt)
        # tgt = self.pos_decoder(tgt)
        
        memory_traj = self.encoder(historic_traj) # [BS, A, H, d_model]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(bs, num_agents, 1, 1) # [BS, A, K, d_model] (k = number of trajectories)
        out_traj = self.decoder (self.query_batches, memory_traj) # [BS, A, K, d_model]
        # ----------------------------------------------------------------------- #
        # Lane step
        lanes_enc = self.process_lanes(lanes=lanes) # [BS, max_lanes_num, num_features]
        lanes_enc = self.lanes_emb(lanes_enc) # [BS, max_lanes_num, num_features]
        lanes_mem = self.lane_encoder(lanes_enc)
        lanes_mem = lanes_mem.unsqueeze(1).repeat(1, num_agents, 1, 1) # [BS, A, max_lanes_num, num_features] -> Homogeinity
        out = self.lane_decoder(out_traj, lanes_mem) # out traj as target lane_mem as memory
        # ----------------------------------------------------------------------- #
        num_traj = out.shape[2] # Number of predictions (K)
        out = out.view (bs, num_agents, num_traj, -1)
        # Get the output i the expected dimensions
        pred: torch.Tensor = self.reg_mlp(out)
        pred = pred.view(bs, num_agents, num_traj, -1, num_features) # [bs, A, K, F, out feats]
        num_traj = pred.shape[1] # Number of predictions (K)
        
        cls_h = self.cls_FFN(out)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf: torch.Tensor = self.cls_opt(cls_h)
        
        return pred, conf


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
        x = x + Variable(self.pe[:x.shape[-2]], requires_grad=False)
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