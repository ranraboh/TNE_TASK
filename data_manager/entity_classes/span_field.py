from typing import Tuple


class SpanField:
    """
        DESCRIPTION: The class describes a span (sequence of tokens) in the text.
        Each span represented by its document index, start and end positions
    """
    def __init__(self, id: str, start_position: int, end_position: int) -> None:  # document id (?)
        """
            DESCRIPTION: The method init the fields/information regarding the span.
            ARGUMENTS:
              - id: identification of the span
              - start_position: the position the span begins in the text.
              - end_position: the position the span ends in the text.
        """
        # Init span fields
        self.id = id
        self.start_position = start_position

        # Adding 1 to the end tokens to account for the non-including nature of index selection
        self.end_position = end_position + 1

    def end_points(self) -> Tuple:
        """
            DESCRIPTION: The method returns a tuple which describes the end points of the span.
            RETURNS: (tuple) The method returns a tuple which contains two entries
            such that the first entry is the start position and the second entry is the end position.
        """
        return self.start_position, self.end_position

    def __str__(self) -> str:
        """
          DESCRIPTION: The method return a string representation of the span
          which includes information regarding the noun phrase: id, start position, end position
          RETURN: (str) string representation of SpanField object.
        """
        return "ID: " + str(self.id) + "\t Positions: (" + str(self.start_position) + ", " + str(self.end_position) + ")"

