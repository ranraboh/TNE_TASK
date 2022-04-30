import torch.nn as nn
import torch.nn.functional as F
from model.graph_modules.gcn_layer import GraphConvolutionLayer
from typing import Dict, List, Optional, Any

class GraphConvolution(nn.Module):
    def __init__(self, config, device_type: Optional[str] = "cuda"):
        super(GraphConvolution, self).__init__()
        nhid = config['hidden_dim']
        self.gc_inner_layers = [ GraphConvolutionLayer(nhid, nhid, bias=True, device_type=device_type) for _ in range(config['nof_layers'] - 2) ]
        self.gc_in = GraphConvolutionLayer(config['input_dim'], nhid, bias=True, device_type="cpu")
        self.gc_out = GraphConvolutionLayer(nhid, config['output_dim'], bias=True, device_type=device_type)
        self.dropout = config['dropout']

    def forward(self, x, adj):
        x = F.relu(self.gc_in(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc_inner_layer in self.gc_inner_layers:
            x = F.relu(gc_inner_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        out = self.gc_out(x, adj)
        return out
