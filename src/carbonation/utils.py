import re
from typing import List

INT = re.compile(r"^\d+$")
COL_RANGE = re.compile(r"^\d+-\d+$")
ITERATOR = re.compile(r"(?P<start>\d+)\+\+(?P<step>\d+)<(?P<stop>\d+)")


def expand_column_list(spec: str) -> List[int]:
    """

    Examples:
        1-2 --> [1, 2]
        2-1 --> [2, 1]
        1,2 --> [1, 2]
        1,2,3-4 --> [1, 2, 3, 4]

    """
    out: List[int] = []
    for part in spec.split(","):
        if INT.match(part):
            a = int(part)
            out.append(a)
        elif COL_RANGE.match(part):
            a, b = part.split("-")
            a, b = int(a), int(b)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        elif m := ITERATOR.match(part):
            start = int(m.group("start"))
            stop = int(m.group("stop"))
            step = int(m.group("step"))
            out.extend(list(range(start, stop + 1, step)))

        else:
            msg = f"Could not expand column list spec {part!r}"
            raise ValueError(msg)
    return out
