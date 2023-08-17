import torch as t
import numpy as np
import pandas as pd
from scipy import signal
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
import torch_geometric as tg


class GNN(t.nn.Module):
    def __init__(self, input_dim, time_dim, feature_dim):
        super(GNN, self).__init__()

        # basic parameters
        self.input_dim = input_dim  # num of nodes 
        self.time_dim = time_dim  # num of timesteps
        self.feature_dim = feature_dim  # num of features

        self.hidden_size = 128
        self.lstm_layers = 2
        self.node_attr_dim = self.time_dim * self.feature_dim
        self.num_edge_features = 1 #+ 1
        self.dropout = 0.3

		# # Node feature encoder
        # self.node_encoder_layer_1 = t.nn.Linear(self.feature_dim, self.hidden_size)
        # self.node_encoder_dropout_1 = t.nn.Dropout(self.dropout)
        # self.node_encoder_activation_1 = t.nn.ReLU()
        # self.node_encoder_layer_2 = t.nn.Linear(self.hidden_size, self.hidden_size)
        # self.node_encoder_dropout_2 = t.nn.Dropout(self.dropout)
        # self.node_encoder_activation_2 = t.nn.ReLU()
        # self.node_encoder_layer_norm = t.nn.LayerNorm([self.hidden_size])

        # LSTM
        self.lstm_encoder = t.nn.LSTM(
            input_size = self.feature_dim, 
            hidden_size = self.hidden_size,
            num_layers = self.lstm_layers, 
            dropout = self.dropout, 
            bidirectional = False,
        )
        self.lstm_layer_norm = t.nn.LayerNorm([self.hidden_size])
        hidden_state = t.zeros(self.lstm_layers, self.hidden_size)
        cell_state = t.zeros(self.lstm_layers, self.hidden_size)
        t.nn.init.xavier_normal_(hidden_state)
        t.nn.init.xavier_normal_(cell_state)
        # If variable batch size is required, 'persistent' should be False
        self.register_buffer('hidden_state', hidden_state, persistent=True)
        self.register_buffer('cell_state', cell_state, persistent=True)

        # GNN
        self.gnn_processor_1 = tg.nn.GATv2Conv(
            in_channels = self.hidden_size,
            out_channels = self.hidden_size,
            edge_dim = self.num_edge_features, 
            dropout = self.dropout,
        )
        # self.gnn_processor_2 = tg.nn.GATv2Conv(
        #     in_channels = self.hidden_size,
        #     out_channels = self.hidden_size,
        #     edge_dim = self.num_edge_features, 
        #     dropout = self.dropout,
        # )
        self.gnn_layer_norm = t.nn.LayerNorm([self.hidden_size])

        # MLP Decoder
        # self.node_decoder_layer_1 = t.nn.Linear(self.hidden_size, self.hidden_size)
        # self.node_decoder_activation_1 = t.nn.ReLU()
        # self.node_decoder_layer_2 = t.nn.Linear(self.hidden_size, self.output_seq_len)
        self.mlp_decoder = t.nn.Linear(self.hidden_size, 1)


    # def node_encoder(self, x):
    #     x = self.node_encoder_layer_1(x)
    #     x = self.node_encoder_dropout_1(x)
    #     x = self.node_encoder_activation_1(x)
    #     x = self.node_encoder_layer_2(x)
    #     x = self.node_encoder_dropout_2(x)
    #     x = self.node_encoder_activation_2(x)
    #     x = self.node_encoder_layer_norm(x)
    #     return x

    def apply_lstm(self, x, hidden):
        x, hidden = self.lstm_encoder(x, hidden)
        x = self.lstm_layer_norm(x)
        return x, hidden
    

    def apply_gnn(self, x, edge_index, edge_attr):
        x = self.gnn_processor_1(x, edge_index, edge_attr)
        # x = self.gnn_processor_2(x, edge_index, edge_attr)
        x = self.gnn_layer_norm(x)
        return x


    def apply_decoder(self, x):
        x = self.mlp_decoder(x)
        return x


    def forward(self, inputs):
        batch_size = inputs.size()[0]
        outputs = t.zeros(batch_size, self.input_dim).type_as(inputs)

        for i in range(batch_size):
            # LSTM
            lstm_inputs = inputs[i].view(self.input_dim, self.time_dim, self.feature_dim)
            lstm_outputs = t.zeros(self.input_dim, self.hidden_size).type_as(inputs)
            for j in range(self.input_dim):
                _, (h_out, _) = self.apply_lstm(lstm_inputs[j], (self.hidden_state, self.cell_state))
                lstm_outputs[j] = h_out[-1].view(-1, self.hidden_size) # (stocks, 128)

            # GNN
            # gnn_inputs = inputs[i].view(self.input_dim, self.node_attr_dim)
            edge_index, edge_attr = self._get_edge_index_and_attr() # (gnn_inputs)
            gnn_outputs = self.apply_gnn(lstm_outputs, edge_index, edge_attr)  # (stocks, 128)

            # Decoder
            # x = self.node_decoder(x)  # MLP node decoder
            decoder_outputs = self.apply_decoder(gnn_outputs)  # (stocks, 1)
            decoder_outputs = decoder_outputs.view(1, self.input_dim)  # (1, stocks)
            outputs[i] = decoder_outputs

        return outputs
    
    
    def _get_edge_index_and_attr(self, X=None):
        edge_index = []  # [2, Number of edges]
        edge_attr = []  # [Number of edges, Edge Feature size]
    
        # # Calculate cross-correlation for dynamically creating edges
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

        # X = pd.DataFrame(t.Tensor.cpu(X).numpy())
        # EPSILON = 1e-10
        # for node_i in range(self.input_dim):
        #     for node_j in range(self.input_dim):
        #         flag = False
        #         attr = []
        #         relative_pos = node_j - node_i
        #         attr.append(relative_pos)  # Add relative position
        #         # for k in range(self.feature_dim):  # Indicators
        #         for k in range(1):  # Price
        #             coin_i = X.iloc[node_i, k::self.time_dim]
        #             coin_j = X.iloc[node_j, k::self.time_dim]
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

        # # Fully-connected graph if no edges
        # if len(edge_index) == 0:
        #     for node_i in range(self.input_dim):
        #         for node_j in range(self.input_dim):
        #             edge_index.append([node_i, node_j])
        #             attr = []
        #             attr.append(node_j - node_i)  # Add relative position
        #             attr.append(X.iloc[node_i, -self.feature_dim] - X.iloc[node_j, -self.feature_dim])  # Add relative price
        #             edge_attr.append(attr)

        # Fully-connected graph with learnable edge weights
        for node_i in range(self.input_dim):
            for node_j in range(self.input_dim):
                edge_index.append([node_i, node_j])
                attr = []
                attr.append(node_j - node_i)  # Add relative position
                edge_attr.append(attr)

        edge_index = t.tensor(np.asarray(edge_index), dtype=t.long).t().to('cuda:0')
        edge_attr = t.tensor(np.asarray(edge_attr), dtype=t.float).to('cuda:0')
        return edge_index, edge_attr