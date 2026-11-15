from weathercam.core.text_metrics import levenshtein


def plate_distance(reading: str, query: str) -> int:
    return levenshtein(reading, query)


def match_plate(reading: str, query: str, max_edits: int) -> bool:
    if max_edits < 0:
        raise ValueError(f"max_edits must be >= 0, got {max_edits}")
    if reading == "":
        return False
    return plate_distance(reading, query) <= max_edits
