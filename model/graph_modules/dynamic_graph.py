import torch
from model.graph_modules.graph_convolution import GraphConvolution
from typing import Dict, List, Optional, Any
from model.modules.mlp_module import Multilayer_Classifier

class DynamicGraphModule(torch.nn.Module):
    def __init__(self, config, device_type: Optional[str] = "cuda"):
        super(DynamicGraphModule, self).__init__()

        # dynamic graph hyper parameters
        span_compression_config = config['span_compression']
        link_classification_config = config['link_classification']
        graph_convolution_config = config['graph_convolution']
        self.device_type = device_type

        # Sparsity theresehold
        self.sparsity_thresehold = torch.ones(1, device=device_type) * 1.5
        self.min_sparsity_thresehold = torch.ones(1, device=device_type) * -1.5
        self.no_link = torch.zeros(1, device=device_type)

        # Init the inner modules of the dynamic graph
        self.span_compression = Multilayer_Classifier(mlp_config=span_compression_config, output_probs=False)
        self.link_correlation = torch.nn.Linear(link_classification_config['input_dim'], link_classification_config['output_dim'], bias=True, device=self.device_type)
        self.graph_convolution = GraphConvolution(config=graph_convolution_config, device_type=device_type)

    def forward(self, spans_embeddings):
        # Learn the connections of the graph which map out the way the information is conveyed between the spans
        spans_embeddings = self.span_compression(spans_embeddings)
        batch_size, nof_spans, features_dim = spans_embeddings.shape
        links_mat = torch.cat([spans_embeddings.repeat(1, nof_spans, 1),
                               spans_embeddings.repeat(1, 1, nof_spans).view(batch_size, -1, features_dim)], dim=-1)
        x = self.link_correlation(links_mat)[0, :, 1].view(nof_spans, nof_spans)

        # The evaluated scores are normalized with zero mean and unit variance
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        links_correlation = ((x - mean)/ std)

        # Apply sparsity scheme where scores that their standard deviations are 1.5 above or -1.5
        # below the mean are modeled in the graph and the others are left out and do not include in the aggregation process.
        links_adjancy_matrix = torch.where(links_correlation > self.sparsity_thresehold, links_correlation, self.no_link)
        links_adjancy_matrix = torch.where(links_adjancy_matrix < self.min_sparsity_thresehold, links_adjancy_matrix, self.no_link)

        # The GNN is applied which used to share information among spans
        # Each node considers its immediate local neighborhood to update his features.
        output = self.graph_convolution(spans_embeddings.squeeze(0), links_adjancy_matrix)
        return output.unsqueeze(0)