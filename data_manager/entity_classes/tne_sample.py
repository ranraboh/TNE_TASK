from typing import List, Optional
from data_manager.entity_classes.span_field import SpanField
import torch


class TNESample:
    def __init__(self, id: str, tokens: List[str], spans: List[SpanField], links: torch.Tensor, prepositions_labels: torch.Tensor) -> None:
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

    def __str__(self) -> str:
        """
        DESCRIPTION: the method return a string representation of TNESample
        which includes information regarding the sample: id, tokens, spans, prepositions relations between
        noun phrases in text etc.
        RETURN (str): string representation of TNESample object.
        """
        return "ID: " + str(self.id) + "\t Tokens: " + str(self.tokens) + "\t Spans:" + str(self.spans) \
               + "\t Links:" + str(self.links) + "\t Prepositions Relations: " + str(self.prepositions_labels)
