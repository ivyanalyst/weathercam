def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        raise ValueError("reference must not be empty")
    return levenshtein(reference, hypothesis) / len(reference)


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, got {len(references)} and {len(hypotheses)}"
        )
    if not references:
        raise ValueError("references must not be empty")
    total = sum(len(r) for r in references)
    if total == 0:
        raise ValueError("total reference length must be greater than zero")
    edits = sum(levenshtein(r, h) for r, h in zip(references, hypotheses))
    return edits / total


def _check_pair(references: list[str], hypotheses: list[str]) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, got {len(references)} and {len(hypotheses)}"
        )
    if not references:
        raise ValueError("references must not be empty")


def exact_match_rate(references: list[str], hypotheses: list[str]) -> float:
    _check_pair(references, hypotheses)
    return sum(r == h for r, h in zip(references, hypotheses)) / len(references)


def position_accuracy(references: list[str], hypotheses: list[str]) -> list[float]:
    _check_pair(references, hypotheses)
    width = len(references[0])
    if width == 0:
        raise ValueError("references must not be empty strings")
    if any(len(r) != width for r in references):
        raise ValueError("all references must have the same length")
    hits = [0] * width
    for r, h in zip(references, hypotheses):
        for i in range(min(width, len(h))):
            if r[i] == h[i]:
                hits[i] += 1
    return [count / len(references) for count in hits]
