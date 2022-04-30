from torch.nn.utils.rnn import pad_sequence
from model.modules.positional_encoding import PositionalEncoding
import torch



class HierarchicalEncoder(torch.nn.Module):
    def __init__(self, config, device="cuda"):
        super(HierarchicalEncoder, self).__init__()
        self.config = config
        # Context layer which used to gain contextualized representation
        # of each token in the text.
        self.subsequence_levels = config['subsequence_levels']

        # Transformer-based encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config['input_dim'], nhead=config['nof_heads'], dropout=config['dropout'], batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config['nof_layers']) # layers=3, 4

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=config['positional_dim'])
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, tokens_embeddings, spans):
        # Concat each token with a positional encoding
        context_output = tokens_embeddings
        tokens_embeddings = self.positional_encoding(self.dropout(tokens_embeddings))

        # The module hierarchically refines the encoding of the tokens to gain a better representation.
        for subsequence_size in self.subsequence_levels:
            tokens_embeddings_split =  tokens_embeddings.split(subsequence_size, dim=1)
            subsequences_embeddings = []
            for subsequence in tokens_embeddings_split:
                x = self.encoder(subsequence)
                subsequences_embeddings.append(x)
            tokens_embeddings = torch.concat(subsequences_embeddings, dim=1)

        # The representation gained by the hierarchical encoder adds up to
        # the encoding acquired by the pre-trained layer.
        spans_embeddings = []
        for batch_spans in spans:
            for span in batch_spans:
                span_context_output = context_output[span[0]][span[1]:span[2]]
                span_encoder_output = self.encoder(tokens_embeddings[span[0]][span[1]:span[2]].unsqueeze(0)).squeeze(0)
                x1, x2 = span_context_output[0], span_context_output[-1]
                x3, x4 = span_encoder_output[0], span_encoder_output[-1]
                spans_embeddings.append(torch.concat((x1, x2, x3, x4), dim=-1))
        return pad_sequence(spans_embeddings).transpose(0, 1).unsqueeze(0)
