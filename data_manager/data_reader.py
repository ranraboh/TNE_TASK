from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
from data_manager.data_utils import read_data
from data_manager.data_access import DataAccess
import torch

from transformers import AutoTokenizer
from data_manager.entity_classes.span_field import SpanField
from data_manager.entity_classes.tne_sample import TNESample
import sys

class DataReader:
    def __init__(self, prepositions_list: List[str]) -> None:
        """
            DESCRIPTION: The method init a data reader object        """
        # Map between a preposition name to corresponding index
        self.prepositions_counter = len(prepositions_list)
        self.prepositions_dict = {k: v for v, k in enumerate(prepositions_list)}
        self.tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-base-cased')
        self.classes = torch.zeros(26, dtype=torch.long)


        #
        self.no_relation = 0
        self.self_relation = 1

    def load_samples(self, data_path: str) -> Dataset:
        """
            DESCRIPTION: the method used to read/load the data into the memory,
            pre-process the data to fit the expected format for the model.
            RETURN (Dataset): the method returns an Dataset object which contains the list of samples
            of the dataset and provides an API to access the samples.
        """
        # Path of the data
        data_path = data_path

        # Read/Load the dataset from file.
        data = read_data(data_path)

        # Build each sample
        documents = []
        for sample in data:
            document = self.create_sample_instance(sample)
            documents.append(document)

        # Return a DataSet object which provides interface for accessing elements in the dataset.
        return DataAccess(documents)

    def create_sample_instance(self, sample: Dict[str, Any]):
        """
            DESCRIPTION: helper method which gain json-formatted information regarding a document
            in the following format:
            "text": the text/content of the document,
            "tokens": list of the tokens
            "nps": list of the noun phrases spans in the text.
            such that each np is identified by the positions of the first and end tokens.
            "links" (DICT[str, Any]): for each ordered pair of NPs spans that has
        """
        # List which contains the tokens of the document.
        tokens = sample['tokens']

        # Extract noun phrase information and build list of all the spans in the text/document.
        spans = {}
        np_entities = sample['nps']
        for np_id, np_record in np_entities.items():
            first_token_idx = np_record['first_token']
            last_token_idx = np_record['last_token']
            spans[np_id] = SpanField(id=np_id, start_position=first_token_idx, end_position=last_token_idx)  # document id ?

        # Build mapping between an anchor and complement noun phrases to
        # the preposition-relation between them.
        links = defaultdict(dict)
        prepositions_relations = defaultdict(dict)
        np_relations = sample['np_relations']
        for np_relation in np_relations:
            anchor = np_relation['anchor']
            complement = np_relation['complement']
            preposition = np_relation['preposition']
            links[anchor][complement] = 1
            if preposition in self.prepositions_dict:
                preposition_idx = self.prepositions_dict[preposition]
            else:
                self.prepositions_dict[preposition] = self.prepositions_counter
                self.prepositions_counter += 1
                preposition_idx = self.prepositions_counter
            prepositions_relations[anchor][complement] = preposition_idx

        #
        nof_spans = len(np_entities)
        links_vec = torch.zeros(nof_spans, nof_spans, dtype=torch.long, device="cpu")
        preposition_labels = []
        for i in range(0, nof_spans):
            for j in range(0, nof_spans):
                if i == j:
                    links_vec[i][j] = self.no_relation
                    preposition_labels.append(self.self_relation)
                    self.classes[self.self_relation] += 1
                    continue
                ith_span = "np" + str(i)
                jth_span = "np" + str(j)
                if ith_span in prepositions_relations and jth_span in prepositions_relations[ith_span]:
                    preposition_labels.append(prepositions_relations[ith_span][jth_span])
                    self.classes[prepositions_relations[ith_span][jth_span]] += 1
                    links_vec[i][j] = 1
                else:
                    preposition_labels.append(self.no_relation)
                    self.classes[self.no_relation] += 1

        preposition_labels = torch.LongTensor(preposition_labels, device="cpu")
        sample = TNESample(id=sample['id'], tokens=tokens, spans=spans, links=links_vec,
                         prepositions_labels=preposition_labels)
        sample.adjust_spans(self.tokenizer)
        return sample