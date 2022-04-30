from typing import List, Optional
class CoreferenceGroup:
    def __init__(self, id: str, members: List[str], np_type: str) -> None:
        """
            DESCRIPTION: The method init the fields/information regarding the co-reference.
            ARGUMENTS:
              - id: identification of the coreference
              - members: the noun phrases/spans of the coreference
        """
        # Init co-reference fields
        self.id = id
        self.members = members
        self.np_type = np_type