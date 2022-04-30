import torch
from transformers import AutoTokenizer, RobertaTokenizerFast

class TNE_Config:
    """
      DESCRIPTION: contains the generic configuration for the TNE task
      for instance:
        - Train set: The path of file which contains the training samples which used
        to train the model.
        - Evaluation set: The path of files which contains the dev set which
        used to evaluate the model.
        - Prepositions list: list of the valid prepositions that can be preposition-link
        between an anchor noun phrase to complement span.
        The prepositions list is the possible labels of TNE task.
    """
    def __init__(self) -> None:
        # Paths of the files of datasets of TNE task.
        self.train_set = 'data/train.jsonl.gz'
        self.evaluation_set = 'data//dev.jsonl.gz'
        self.test_set = 'data/test_unlabeled.jsonl.gz'

        # Directories paths to save the results/outputs and logs.
        self.output_dir = './results'
        self.logs_dir = './logs'
        self.best_model_path = './save/best_model'
        self.test_output = './results/test.json'

        # Prepositions List - list of the possible prepositions that can be the relation
        # between an anchor noun phrase to complement span.
        # add a 'self' for the preposition list which used when comparing a span to itself, 'self' relation does not include when computing the metrics.
        self.prepositions_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                            'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                            'inside', 'outside', 'into', 'around', 'self']
        self.num_labels = len(self.prepositions_list)
        self.tokenizer_type = "roberta-large"
        self.span_start_token = "<span_start>"
        self.span_end_token = "<span_end>"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.tokenizer_type, add_prefix_space=True)
        self.tokenizer.add_tokens([self.span_start_token, self.span_end_token])

        # Define device type. If available, use cuda to utilize GPUs for computation
        if torch.cuda.is_available:
            self.device = "cuda"
        else:
            self.device = "cpu"
