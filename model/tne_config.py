import torch


class TNE_Config:
    """
      DESCRIPTION: contains the generic configuration for the task
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

        # Directories paths to save the results/outputs and logs.
        self.output_dir = './results'
        self.logs_dir = './logs'

        # Prepositions List - list of the possible prepositions that can be the relation
        # between an anchor noun phrase to complement span.
        self.prepositions_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                            'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                            'inside', 'outside', 'into', 'around']
        self.num_labels = len(self.prepositions_list)

        # Define device type. If available, use cuda to utilize GPUs for computation
        if torch.cuda.is_available:
            self.device = "cuda"
        else:
            self.device = "cpu"
