from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
from data_manager.data_utils import read_data
from data_manager.data_access import DataAccess
import torch

from data_manager.entity_classes.coreference import CoreferenceGroup
from data_manager.entity_classes.span_field import SpanField
from data_manager.entity_classes.tne_sample import TNESample

class DataReader:
    def __init__(self, prepositions_list: List[str], tokenizer) -> None:
        """ init a data reader object """
        # Map between a preposition name to corresponding index
        self.prepositions_counter = len(prepositions_list)
        self.prepositions_dict = {k: v for v, k in enumerate(prepositions_list)}
        self.tokenizer = tokenizer

        # Special labels
        self.no_relation = self.prepositions_dict['no-relation']
        self.self_relation = self.prepositions_dict['self']

    def load_samples(self, data_path: str, test_mode=False) -> Dataset:
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
            document = self.create_sample_instance(sample, test_mode)
            documents.append(document)
        return DataAccess(documents)

    def create_sample_instance(self, sample: Dict[str, Any], test_mode):
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
            spans[np_id] = SpanField(id=np_id, start_position=first_token_idx, end_position=last_token_idx)

        # Extract co-reference clusters information
        coreferences_info = sample['coref']
        coreferences = []
        for coref_info in coreferences_info:
            current_coref_spans = coref_info["members"]
            current_coref = CoreferenceGroup(id=coref_info["id"], members=current_coref_spans, np_type=coref_info["np_type"])
            for span_id in current_coref_spans:
                spans[span_id].coreference = current_coref
            coreferences.append(current_coref)

        # Build mapping between an anchor and complement noun phrases to
        # the preposition-relation between them.
        preposition_labels = None
        links_vec = None
        if not test_mode:
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

            # Create a matrix which includes the preposition for each pair of noun phrases
            nof_spans = len(np_entities)
            preposition_labels = []
            links_vec = torch.eye(nof_spans, dtype=torch.long, device="cpu")
            for i in range(0, nof_spans):
                for j in range(0, nof_spans):
                    if i == j:
                        links_vec[i][j] = self.no_relation
                        preposition_labels.append(self.self_relation)
                        continue
                    ith_span = "np" + str(i)
                    jth_span = "np" + str(j)
                    if ith_span in prepositions_relations and jth_span in prepositions_relations[ith_span]:
                        preposition_labels.append(prepositions_relations[ith_span][jth_span])
                        links_vec[i][j] = 1
                    else:
                        preposition_labels.append(self.no_relation)
            preposition_labels = torch.LongTensor(preposition_labels, device="cpu")

        # Create adjacency matrix for the co-reference graph
        nof_spans = len(np_entities)
        coreference_links = torch.eye(nof_spans, device="cpu")
        for i in range(0, nof_spans):
            for j in range(0, nof_spans):
                ith_span = "np" + str(i)
                jth_span = "np" + str(j)
                if spans[ith_span].coreference == spans[jth_span].coreference and spans[ith_span].coreference is not None:
                    coreference_links[i][j] = 1

        # Create a TNEsample object which contains the information regarding a sample in the dataset
        sample = TNESample(id=sample['id'], tokens=tokens, spans=spans, links=links_vec,
                         prepositions_labels=preposition_labels, coreference_links=coreference_links, test_mode=test_mode)
        sample.process_data(self.tokenizer)
        return sample