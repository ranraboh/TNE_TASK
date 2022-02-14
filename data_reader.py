from data_manager.data_utils import read_data
import tokenizers

class SpanField:
    """ """
    def __init__(self, tokens: list[str], start_position: int, end_position: int):
        # Init span fields
        self.tokens = tokens
        self.start_position = start_position
        self.end_position = end_position


class DataReader:
    def __init__(self):
        self.spans = {}


file = "data/train.jsonl.gz"
data = read_data(file)
print (data)