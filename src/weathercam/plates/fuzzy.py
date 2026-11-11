"""Fuzzy matching of an OCR plate reading against a query plate."""

from weathercam.core.text_metrics import levenshtein


def plate_distance(reading: str, query: str) -> int:
    """Edit distance between an OCR reading and a query plate.

    Characters are compared as they are: no case folding, and the province
    character counts like any other position.
    """
    return levenshtein(reading, query)


def match_plate(reading: str, query: str, max_edits: int) -> bool:
    """True if the reading is within max_edits edits of the query.

    max_edits has no default: the caller states the limit. A negative limit
    raises ValueError. An empty reading never matches.
    """
    if max_edits < 0:
        raise ValueError(f"max_edits must be >= 0, got {max_edits}")
    if reading == "":
        return False
    return plate_distance(reading, query) <= max_edits
