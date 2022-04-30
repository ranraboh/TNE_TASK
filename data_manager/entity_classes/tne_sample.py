from typing import List, Optional
from data_manager.entity_classes.span_field import SpanField
import torch

class TNESample:
    def __init__(self, id: str, tokens: List[str], spans: List[SpanField], links: torch.Tensor, prepositions_labels: torch.Tensor, coreference_links, test_mode=False) -> None:
        """
            DESCRIPTION: The method used to init the fields/information regarding a sample in dataset.
            ARGUMENTS:
              - id (str): a unique identifier for sample, beginning with r and followed by number.
              - tokens (List[str]): list which contains the tokens of the document.
              - spans (List[SpanField]): list of all spans/noun phrases in the text.
              - links (torch.Tensor): indicate for each pair of noun phrases, whether they are linked or not.
                links tensor dim: [ NOF_SPANS x NOF_SPANS ]
              - prepositions_labels (torch.Tensor): determine the connecting preposition entity for each pair of noun phrases.
                tensor dim: [ NOF_SPANS x NOF_SPANS ]
        """
        # Unique identifier of the sample
        self.id = id

        # List which contains the tokens of the document.
        self.tokens = tokens

        # Information regarding the spans/noun phrases in the documents.
        self.spans_quantity = len(spans)
        self.spans = spans
        self.spans_range = None

        # Indicate for each pair of noun phrases, whether they are linked or not.
        self.links = links
        self.coreference_links = coreference_links

        # Determine the connecting preposition entity for each pair of noun phrases
        self.prepositions_labels = prepositions_labels

        # The co-reference relations between all NPs in the text.
        # for each two nps x, y with a co-reference relation
        # such that x and y belongs to the same co-reference cluster then
        # if there is prepositional relation (z, prep, x), It can be inferred
        # that the following relation (z, prep, y) holds as well.
        self.co_references = None  # Not Used yet

        # Additional meta-data
        self.metadata = None
        self.test_mode = test_mode

        # Init concrete relation data which contains the labels for pair of NP span that has a
        # concrete preposition relation between them and their corresponding index
        if not self.test_mode:
            self.init_concrete_relations()

    def get_spans_range(self, batch_idx) -> torch.Tensor:
        """
            DESCRIPTION: The method returns the end points/range of the spans of the sample/document.
            The information is returns as tensor of size [ SPANS_QUANTITY x 3 ]
            such that each row contains the end points of a span in the document.
            ARGUMENTS:
              - device_type: type of device
        """
        spans_range = torch.zeros(self.spans_quantity, 3, dtype=torch.long)
        for i, span in enumerate(self.spans.values()):
            spans_range[i] = span.get_end_points(batch_idx)
        return spans_range

    def process_data(self, tokenizer):
        """
            The method process the data to fits to the model.
            Add start and end tokens
        """
        span_start_token = "<span_start>" # extract from tokenizer
        span_end_token = "<span_end>"
        for i, span in enumerate(self.spans.values()):
            span.adjust_end_points(span.start_position + 2 * i, span.end_position + 2 * (i + 1))
            self.tokens.insert(span.start_position, span_start_token)
            self.tokens.insert(span.end_position - 1, span_end_token)

        subwords_tokens = torch.zeros(len(self.tokens) + 1, dtype=torch.long)
        subwords_tokens[0] = 1
        for id, word in enumerate(self.tokens):
            subword_count = len(tokenizer([word])['input_ids'][0]) - 2
            subwords_tokens[id + 1] = subword_count
        subwords_tokens = torch.cumsum(subwords_tokens, dim=0)
        for i, span in enumerate(self.spans.values()):
            span.adjust_end_points(subwords_tokens[span.start_position], subwords_tokens[span.end_position])

    def init_concrete_relations(self):
        concrete_labels = []
        concrete_idx = []
        for idx, label in enumerate(self.prepositions_labels):
            if label == 0:
                continue
            concrete_labels.append(label)
            concrete_idx.append(idx)
        self.concrete_labels = torch.LongTensor(concrete_labels)
        self.concrete_idx = torch.LongTensor(concrete_idx)

    def __str__(self) -> str:
        """
        DESCRIPTION: the method return a string representation of TNESample
        which includes information regarding the sample: id, tokens, spans, prepositions relations between
        noun phrases in text etc.
        RETURN (str): string representation of TNESample object.
        """
        return "ID: " + str(self.id) + "\t Tokens: " + str(self.tokens) + "\t Spans:" + str(self.spans) \
               + "\t Links:" + str(self.links) + "\t Prepositions Relations: " + str(self.prepositions_labels)