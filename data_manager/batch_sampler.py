from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional
from data_manager.entity_classes.tne_sample import TNESample
from data_manager.entity_classes.tne_batch import TNEBatch
from transformers import AutoTokenizer
import torch


class Batch_Sampler:
    def __init__(self, tokenizer_type: str, device_type: Optional[str] = "cuda"):
        """
            DESCRIPTION: The method init the fields/information crucial for Batch Sampler
            ARGUMENTS:
              - tokenizer_type (str): type of tokenizer used to prepare the inputs for
                the model.
              - device_type (Optional[str]): device type (options: cuda, cpu)
        """
        # Tokenizer used to map each token string to its index in the token space
        # and prepare the inputs for context model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

        # Device type (options: cuda, cpu)
        self.device_type = device_type

    def __call__(self, batch: List[TNESample]):
        """
            DESCRIPTION: The method gathered the data of a batch of samples.
            Used by the DataLoader to collate individual fetched data samples into batches.
            ARGUMENTS:
              - batch (List[TNESample]): list of all the samples in the batch
        """
        # Number of samples in the batch
        batch_size = len(batch)

        # Aggregate the data of each document in the batch.
        batch_spans = []
        tokenized_data = []
        batch_links = []
        batch_labels = []
        for batch_idx, sample in enumerate(batch):
            tokenized_data.append(sample.tokens)
            batch_spans.append(sample.get_spans_range(batch_idx))
            batch_links.append(sample.links.view(-1))
            batch_labels.append(sample.prepositions_labels)

        # Use the tokenizer to map each token string to its index in the token space
        # and prepare the inputs for context model.
        tokens = self.tokenizer(tokenized_data, padding=True, is_split_into_words=True, return_tensors="pt")
        input_ids = tokens['input_ids']

        # create random array of floats with equal dimensions to input_ids tensor
        # to determine the tokens to mask.
        rand = torch.rand(input_ids.shape)

        # create mask array
        mask_arr = (rand < 0.15) * (input_ids != 101) * \
                   (input_ids != 102) * (input_ids != 0)

        for i in range(input_ids.shape[0]):
            masked_tokens = torch.flatten(mask_arr[i].nonzero()).tolist()
            input_ids[i, masked_tokens] = 103
        tokens['input_ids'] = input_ids

        # Contains all the spans in the batch by its first and last positions.
        # dim: [ BATCH_SIZE * MAX_NOF_SPANS_FOR_DOCUMENT_IN_BATCH x 2 ]
        spans = pad_sequence(batch_spans).view(batch_size, -1, 3)

        # Gather the information of all the relations between noun phrases of the documents in the batch.
        # tensor dim: [ BATCH_SIZE x NOF_SPANS x NOF_SPANS ]
        links = pad_sequence(batch_links).transpose(0, 1).view(batch_size, -1)
        prepositions_labels = pad_sequence(batch_labels).transpose(0, 1).reshape(-1)

        # Build a TNEBatch object with the batch data.
        batch_data = TNEBatch(tokens, spans, links, prepositions_labels)
        output_dict = dict(input=batch_data, labels=batch_data.preposition_labels)
        return output_dict

