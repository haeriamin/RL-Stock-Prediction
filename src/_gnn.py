import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATv2Conv#, SAGPooling, GCNConv, global_mean_pool , global_max_pool, BatchNorm

import capsule_layer as CL


class CapsuleLinear(nn.Module):
    r"""Applies a linear combination to the incoming capsules

     Args:
         out_capsules (int): number of output capsules
         in_length (int): length of each input capsule
         out_length (int): length of each output capsule
         in_capsules (int, optional): number of input capsules
         share_weight (bool, optional): if True, share weight between input capsules
         routing_type (str, optional): routing algorithm type
            -- options: ['dynamic', 'k_means']
         num_iterations (int, optional): number of routing iterations
         squash (bool, optional): squash output capsules or not, it works for all routing
         kwargs (dict, optional): other args:
            - similarity (str, optional): metric of similarity between capsules, it only works for 'k_means' routing
                -- options: ['dot', 'cosine', 'tonimoto', 'pearson']

     Shape:
         - Input: (Tensor): (N, in_capsules, in_length)
         - Output: (Tensor): (N, out_capsules, out_length)

     Attributes:
         if share_weight:
            - weight (Tensor): the learnable weights of the module of shape
              (out_capsules, out_length, in_length)
        else:
            -  weight (Tensor): the learnable weights of the module of shape
              (out_capsules, in_capsules, out_length, in_length)

     Examples::
         >>> import torch
         >>> from capsule_layer import CapsuleLinear
         >>> m = CapsuleLinear(3, 4, 5, 6, share_weight=False, routing_type='dynamic', num_iterations=50)
         >>> input = torch.rand(2, 6, 4)
         >>> output, prob = m(input)
         >>> print(output.size())
         torch.Size([2, 3, 5])
         >>> print(prob.size())
         torch.Size([2, 3, 6])
     """

    def __init__(self, out_capsules, in_length, out_length, in_capsules=None, share_weight=True,
                 routing_type='k_means', num_iterations=3, squash=False, **kwargs):
        super(CapsuleLinear, self).__init__()
        if num_iterations < 1:
            raise ValueError('num_iterations has to be greater than 0, but got {}.'.format(num_iterations))

        self.out_capsules = out_capsules
        self.in_capsules = in_capsules
        self.share_weight = share_weight
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.squash = squash
        self.kwargs = kwargs

        if self.share_weight:
            if in_capsules is not None:
                raise ValueError('Expected in_capsules must be None.')
            else:
                self.weight = Parameter(torch.Tensor(out_capsules, out_length, in_length))
        else:
            if in_capsules is None:
                raise ValueError('Expected in_capsules must be int.')
            else:
                self.weight = Parameter(torch.Tensor(out_capsules, in_capsules, out_length, in_length))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        return CL.capsule_linear(input, self.weight, self.share_weight, self.routing_type, self.num_iterations,
                                 self.squash, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.time_step = time_step
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight,dim=-1)
        attention_probs = torch.transpose(attention_probs,2,1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = 1,
            batch_first = True,
        )
        self.attention_block = AttentionBlock(
            time_step,
            hidden_dim
        ) 
        self.dropout = nn.Dropout(0.2)
        self.hidden_dim = hidden_dim
    
    def forward(self, seq):
        # input : torch.tensor (batch, time_step, input_dim)
        seq_vector,_ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        attention_vec = attention_vec.view(-1, 1, self.hidden_dim) # prepare for concat
        return attention_vec


class CapsGATattentionGRU(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(CapsGATattentionGRU, self).__init__()

        outer_edge = np.ones(shape=(2, input_dim**2))
        count = 0
        for i in range(input_dim):
            for j in range(input_dim):
                outer_edge[0][count] = i
                outer_edge[1][count] = j
                count += 1

        # basic parameters
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.outer_edge = outer_edge
        self.batch = 1
        self.inner_edge = torch.tensor(outer_edge, dtype=torch.int64).to('cuda:0')

        # GRU, TODO: Replace it with LSTM
        self.temporal_encoder = nn.GRU(
            self.input_dim * self.hidden_dim,
            self.input_dim * self.hidden_dim,
            num_layers = 2,
            bidirectional = False,
        )

        # Attention
        self.attention = AttentionBlock(self.time_step, self.hidden_dim)

        # self.TBNLayer = torch.nn.BatchNorm1d(
        #     time_step,
        #     track_running_stats=False,
        # )

        # self.encoder_list = SequenceEncoder(
        #     hidden_dim,
        #     time_step,
        #     input_dim,
        # ) 

        # GNN
        self.inner_gat0 = GATv2Conv(self.hidden_dim, self.hidden_dim)
        self.inner_gat1 = GATv2Conv(self.hidden_dim, self.hidden_dim)
        
        # Capsule
        self.caps_module = CapsuleLinear(
            out_capsules = self.input_dim,
            in_length = 2 * self.hidden_dim,
            out_length = self.hidden_dim,
            in_capsules = None,
            routing_type = 'dynamic',
            num_iterations = 3,
        )

        self.fusion = nn.Linear(hidden_dim, input_dim)


    def forward(self, inputs):
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0)
        
        embedding, _ = self.temporal_encoder(inputs.view(-1, self.time_step, self.input_dim * self.hidden_dim))

        att_vector, _ = self.attention(torch.tanh(embedding)) # (100, dim)

        batch = att_vector.shape[0]

        att_vector = torch.tanh(att_vector.view(-1, self.input_dim, self.hidden_dim))
        x = att_vector.view(-1, self.hidden_dim)

        if self.batch != batch:
            outer_edge = self.outer_edge
            for i in range(1, self.batch):
                outer_edge2 = outer_edge + self.input_dim
                outer_edge = np.concatenate((outer_edge, outer_edge2), axis=1)

            self.inner_edge = torch.tensor(outer_edge, dtype=torch.int64).to('cuda:0')
            self.batch = batch

        # inner graph interaction
        inner_graph_embedding = torch.tanh(self.inner_gat0(x, self.inner_edge))
        inner_graph_embedding0 = torch.tanh(self.inner_gat1(inner_graph_embedding, self.inner_edge.view(2, -1)))
        inner_graph_embedding = torch.add(inner_graph_embedding, inner_graph_embedding0)
        inner_graph_embedding = inner_graph_embedding.view(-1, self.input_dim, self.hidden_dim)

        # fusion 
        fusion_vec = torch.cat((att_vector, inner_graph_embedding), dim=-1)
        caps_out, _ = self.caps_module(fusion_vec)
        out_vec = torch.tanh(self.fusion(torch.tanh(caps_out)))

        return out_vec
