from typing import Dict, List, Any
import json
import gzip


def read_data(file_path: str) -> List[Dict[str, Any]]:
    """
    DESCRIPTION: The method receives a path of a data file.
    which contains set of documents such that a document is represented as a json format.
    and reads/loads the data into a list.
    ARGUMENTS: 
      - file_path (str):  path of the file which contains the data
    RETURN: returns set of Json-formatted elements such that each element
    describes a document.
    """
    with gzip.open(file_path, 'rb') as zip_file:
        data = zip_file.readlines()
        data = [json.loads(x) for x in data]
        return data
