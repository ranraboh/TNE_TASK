import torch


class TNEBatch:
    def __init__(self, tokens: torch.Tensor, spans: torch.Tensor, links: torch.Tensor,
                 preposition_labels: torch.Tensor, coreference_links, concrete_labels, concrete_idx, test_mode=False):
        """
              DESCRIPTION: The method used to init the fields/information regarding a batch of samples from dataset.
              ARGUMENTS:
                - tokens (torch.Tensor): contains the tokenized data for all the documents in the batch.
                - spans (torch.Tensor): contains all spans/noun phrases from batch in the text.
                  such that each span represented by its start and end positions.
                - links (torch.Tensor): indicate for each pair of noun phrases in the documents in the batch,
                  whether they are linked or not.
                - prepositions_labels (torch.Tensor): Aggragete all the relations between noun phrases in
                  the samples of the batch into a single tensor.
                  determine the connecting preposition entity for each pair of noun phrases in batch.
        """

        # Number of documents/samples in the batch
        self.batch_size = len(tokens)
        self.test_mode = test_mode

        # Contains the tokenized data for all the documents in the batch
        # dim: [ BATCH_SIZE x MAX_TOKENS_FOR_DOCUMENT_IN_BATCH ]
        self.tokens = tokens

        # Contains all spans/noun phrases from batch in the text.
        # dim: [ BATCH_SIZE x MAX_NOF_SPANS_FOR_DOCUMENT_IN_BATCH x 2 ]
        self.spans = spans

        # Indicate for each pair of noun phrases in the documents in the batch,
        # whether they are linked or not.
        # links tensor dim: [ BATCH_SIZE x NOF_SPANS x NOF_SPANS ]
        self.links = links

        # Determine the connecting preposition entity for each pair of noun phrases in batch.
        # tensor dim: [ BATCH_SIZE x NOF_SPANS x NOF_SPANS ]
        self.preposition_labels = preposition_labels
        self.coreference_links = coreference_links

        # Labels for pair of NP span that has a concrete preposition relation between them and their corresponding index
        self.concrete_labels = concrete_labels
        self.concrete_idx = concrete_idx

    def __str__(self) -> str:
        """
          DESCRIPTION: The method return a string representation of TNEBatch
          which includes information regarding a batch: id, tokens, spans, links etc.
          RETURN (str): string representation of TNEBatch object.
        """
        return "Tokens: " + str(self.tokens) + "\t Spans:" + str(self.spans) + "\t Links:" + str(
            self.links) + "\t Prepositions Relations: " + str(self.preposition_labels)
    
    def to(self, device_type: str) -> None:
        """
            DESCRIPTION: convert the data regarding the batch to a specific device.
            as cuda or cpu.
        """
        self.tokens = self.tokens.to(device_type)
        self.spans = self.spans.to(device_type)
        self.coreference_links = self.coreference_links.to(device_type)
        if not self.test_mode:
          self.preposition_labels = self.preposition_labels.to(device_type)
          self.concrete_labels = self.concrete_labels.to(device_type)
          self.concrete_idx = self.concrete_idx.to(device_type)
