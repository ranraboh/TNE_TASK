import torch

from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, TFBertModel

from model.hyper_parameters import HyperParameters


class TNEModel(torch.nn.Module):
    def __init__(self, hyper_parameters):
        super(TNEModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #self.span_bert = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
        #self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
        self.hyper_parameters = hyper_parameters

    def forward(self, sentence, eval_mode=False):
        x = self.span_bert(sentence)
        return x


model = TNEModel(hyper_parameters=None)
model("ran raboh is fat")