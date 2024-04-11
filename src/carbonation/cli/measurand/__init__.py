import argparse
import fileinput
import re
from itertools import zip_longest
from typing import Iterable, List

import numpy as np

from carbonation.cli.utils import (
    make_passthru_parser,
    parse_data_to_ndarray,
)
from carbonation.measurand import Measurand, make_measurand
from carbonation.measurand.utils import size_to_uint


def grouper(iterable: Iterable, n: int):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)


RE_INT = re.compile(r"^\d+$")
RE_RANGE = re.compile(r"^\d+-\d+$")
RE_UNBOUNDED = re.compile(r"^\d+-")


class ColumnSpec:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        if RE_INT.match(spec):
            spec = int(spec) - 1
            self.slice = slice(spec, spec + 1)
        elif RE_RANGE.match(spec):
            a, b = spec.split("-")
            a = int(a) - 1
            b = int(b) - 1
            if a <= b:
                self.slice = slice(a, b + 1)
            else:
                self.slice = slice(a, b - 1, -1)
        elif RE_UNBOUNDED.match(spec):
            a, b = spec.split("-")
            a = int(a) - 1
            self.slice = slice(a, a + 1)
        else:
            msg = f"column spec {spec!r} not valid"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(spec={self.spec!r})"


def format_float(value: np.floating) -> str:
    return f"{value: 13.6e}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="meas")
    parser.add_argument("-b", "--bits", type=int, default=8, help="input bits per word")
    parser.add_argument(
        "-w", "--width", type=int, default=None, help="width of column-spec columns"
    )
    parser.add_argument(
        "-c", "--chunk-size", type=int, default=1000, help="processing chunk size"
    )
    parser.add_argument(
        "--passthru",
        "-p",
        type=int,
        default=1,
        help="number of columns to pass thru to output",
    )
    parser.add_argument("measurand", type=str, nargs="+", help="measurand definition")
    args = parser.parse_args()

    if args.width is None:
        width = len(f"{2**args.bits-1}")
    else:
        width = args.width

    measurands: List[Measurand] = []
    for spec in args.measurand:
        try:
            measurands.append(ColumnSpec(spec))
        except ValueError:
            if ";" not in spec:
                spec = spec.replace("/", ";")
            measurands.append(make_measurand(spec, word_size=args.bits))

    line_parser = make_passthru_parser(args.passthru)
    dtype = size_to_uint(args.bits)

    # window = []
    for line in fileinput.input("-"):
        passthru, rest = line_parser(line)
        rest = rest.strip().split()
        print(passthru, end=" ")

        data = parse_data_to_ndarray([rest], dtype=dtype)

        # data = parse_data_to_pa_table(rest, pa.uint8())
        # print(data)
        # data = pa.Table.from_arrays([data], names=[str(i) for i in range(len(data))])
        # data = np.array([int(x) for x in rest], dtype="u1")
        # data = pa.array([int(x) for x in rest], pa.uint8())

        for m in measurands:
            if isinstance(m, ColumnSpec):
                for value in rest[m.slice]:
                    print(f"{value:>{width}}", end=" ")
            elif isinstance(m, Measurand):
                m_width = len(f"{2**m.size-1}")
                value = m.build(data)[0]
                if np.issubdtype(value, np.floating):
                    print(f"{value:13.6e}", end=" ")
                else:
                    print(f"{value:>{m_width}}", end=" ")
                # print(value, end=" ")
                # print(type(value))
            else:
                raise TypeError
            # print(value, end=" ")
        print()
