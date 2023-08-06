import torch as t
import numpy as np
# import pandas as pd
import torch.nn as nn
# from scipy import signal
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
import torch_geometric as tg


class GNN(nn.Module):
    def __init__(self, input_dim, time_dim, feature_dim):
        super(GNN, self).__init__()

        # basic parameters
        self.input_dim = input_dim  # num of nodes 
        self.time_dim = time_dim  # num of timesteps
        self.feature_dim = feature_dim  # num of features

        # TODO
        self.node_attr_dim = self.time_dim * self.feature_dim
        self.num_edge_features = 0

		# # Node feature encoder
        # self.node_encoder_layer_1 = t.nn.Linear(self.feature_dim, self.hidden_size)
        # self.node_encoder_dropout_1 = t.nn.Dropout(self.dropout)
        # self.node_encoder_activation_1 = t.nn.ReLU()
        # self.node_encoder_layer_2 = t.nn.Linear(self.hidden_size, self.hidden_size)
        # self.node_encoder_dropout_2 = t.nn.Dropout(self.dropout)
        # self.node_encoder_activation_2 = t.nn.ReLU()
        # self.node_encoder_layer_norm = t.nn.LayerNorm([self.hidden_size])

        # Processor
        self.processor_gnn = tg.nn.GATv2Conv(self.node_attr_dim, self.node_attr_dim)
        # self.inner_gat0 = tg.nn.GATv2Conv(self.hidden_size, self.hidden_size)
        # self.inner_gat1 = tg.nn.GATv2Conv(self.hidden_size, self.hidden_size)

        # # Node label decoder
        # self.node_decoder_layer_1 = t.nn.Linear(self.hidden_size, self.hidden_size)
        # self.node_decoder_activation_1 = t.nn.ReLU()
        # self.node_decoder_layer_2 = t.nn.Linear(self.hidden_size, self.output_seq_len)

        self.fusion = nn.Linear(self.node_attr_dim, 1)


    # def node_encoder(self, x):
    #     x = self.node_encoder_layer_1(x)
    #     x = self.node_encoder_dropout_1(x)
    #     x = self.node_encoder_activation_1(x)
    #     x = self.node_encoder_layer_2(x)
    #     x = self.node_encoder_dropout_2(x)
    #     x = self.node_encoder_activation_2(x)
    #     x = self.node_encoder_layer_norm(x)
    #     return x


    def processor(self, x, edge_index, edge_attr):
        x = self.processor_gnn(
            x = x,
            edge_index = edge_index,
            edge_attr = None,
        )
        return x


    # def node_decoder(self, x):
    #     x = self.node_decoder_layer_1(x)
    #     x = self.node_decoder_activation_1(x)
    #     x = self.node_decoder_layer_2(x)
    #     return x


    def forward(self, inputs):
        batch_size = inputs.size()[0]

        # # For LSTM
        # inputs = inputs.view(self.input_dim, self.time_dim, self.feature_dim)

        outputs = t.zeros(batch_size, self.input_dim).type_as(inputs)
        
        for i in range(batch_size):
            # For GNN: nodes and their attributes
            x = inputs[i].view(self.input_dim, self.node_attr_dim)
            # For GNN: edges and their attributes
            edge_index, edge_attr = self._get_edge_index_and_attr(x)

            # x = self.node_encoder(x)  # MLP node encoders
            x = self.processor(x, edge_index, edge_attr)  # GNN node and edge processor
            # x = self.node_decoder(x)  # MLP node decoder
            x = self.fusion(x)
            x = x.view(1, self.input_dim)
            outputs[i] = x

        return x
    
    
    def _get_edge_index_and_attr(self, X):
        # def xcorr(norm_signal_1, norm_signal_2):
        #     xcorr = signal.correlate(norm_signal_1, norm_signal_2)  # XCorr
        #     lags = signal.correlation_lags(len(norm_signal_1), len(norm_signal_2))  # Lags
        #     xcorr = xcorr[~np.isnan(xcorr)]  # Sometimes it is all NANs
        #     if xcorr.size != 0:
        #         max_xcorr = np.max(xcorr)  # Max xcorr
        #         phase_shift = lags[np.argmax(xcorr)]  # Phase shift
        #         return max_xcorr, phase_shift
        #     else:
        #         return 0, 0

        edge_index = []  # [2, Number of edges]
        edge_attr = []  # [Number of edges, Edge Feature size]

        # # Calculate cross-correlation
        # X = pd.DataFrame(t.Tensor.cpu(X).numpy())
        # EPSILON = 1e-10
        # for node_i in range(self.input_dim):
        #     for node_j in range(self.input_dim):
        #         flag = False
        #         attr = []

        #         relative_pos = node_j - node_i
        #         attr.append(relative_pos)  # Add relative position

        #         for k in range(self.node_attr_dim):  # Indicators
        #             coin_i = X.iloc[:, node_i * self.node_attr_dim + k].copy()  # TODO
        #             coin_j = X.iloc[:, node_j * self.node_attr_dim + k].copy()  # TODO
        #             mean_i, mean_j = np.mean(coin_i), np.mean(coin_j)
        #             std_i, std_j = np.std(coin_i), np.std(coin_j)
        #             norm_i = (coin_i - mean_i) / (std_i + EPSILON)
        #             norm_j = (coin_j - mean_j) / (std_j + EPSILON)
        #             max_xcorr, phase_shift = xcorr(norm_i, norm_j)  # Phase shift in days
        #             # attr.append(np.round(max_xcorr))  # Add max cross correlation (not good)
        #             attr.append(-phase_shift)	# Add phase shift

        #             if phase_shift < 0:
        #                 flag = True

        #         if flag:
        #             edge_index.append([node_i, node_j])
        #             edge_attr.append(attr)

        # Fully-connected graph (for now)
        for node_i in range(self.input_dim):
            for node_j in range(self.input_dim):
                edge_index.append([node_i, node_j])

        edge_index = t.tensor(np.asarray(edge_index), dtype=t.long).t().to('cuda:0')
        edge_attr = t.tensor(np.asarray(edge_attr), dtype=t.float).to('cuda:0')

        return edge_index, edge_attr