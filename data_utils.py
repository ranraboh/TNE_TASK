import json
import gzip
from tokenizers import Tokenizer


def read_data(file_path: str) -> list:
    """

    :param file_path: (str) the path of file to read
    :return:
    """
    with gzip.open(file_path, 'rb') as zip_file:
        data = zip_file.readlines()
        data = [json.loads(x) for x in data]
        return data


file = "data/train.jsonl.gz"
read_data(file)
