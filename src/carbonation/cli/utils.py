import re
from typing import Callable, List, Literal, Protocol, Tuple, Union

import numpy as np
import pyarrow as pa

UIntDtype = Literal["u1", "u2", "u4", "u8"]


# def parse_line(
#     line: str,
#     dtype: UIntDtype,
#     pass_: int,
# ) -> Tuple[datetime.datetime, np.ndarray]:
#     parts = line.strip().split()
#     passing, values = parts[0:pass_], parts[pass_:]
#     values = np.array([int(x) for x in values], dtype=dtype)
#     return passing, values


class LineParser(Protocol):
    def __call__(self, passthru: int, dtype: UIntDtype) -> Tuple[str, np.ndarray]: ...


def make_passthru_parser(passthru: int) -> Callable[[str], Tuple[str, str]]:
    """Creates a callable that returns two strings:
    - one containing the first N words as it appears in the original string
    - one containing the rest of the string
    """
    pattern = r"^\s*(" + r"\s+".join([r"\S+" for _ in range(passthru)]) + r")"
    pattern = re.compile(pattern)

    def line_parser(line: str) -> Tuple[str, np.ndarray]:
        if m := pattern.match(line):
            passthru = m.groups()[0]
            rest = line[m.end() :]
            return passthru, rest

    return line_parser


def parse_data_to_ndarray(values: List[List[str]], dtype=UIntDtype) -> np.ndarray:
    values = [int(val) for row in values for val in row]
    return np.array(values, dtype=dtype)


def parse_data_to_pa_table(
    values: List[List[str]], dtype: Union[pa.uint8, pa.uint16, pa.uint32, pa.uint64]
) -> pa.array:
    values = [[int(x) if x != "?" else None] for x in values]
    print("values =", values)
    arrays = [pa.array(x, dtype) for x in values]
    print("arrays =", arrays)
    return pa.Table.from_arrays(arrays, names=[str(i) for i in range(len(arrays))])
