import argparse
import fileinput
import re
from itertools import zip_longest
from typing import Iterable, List

import numpy as np
import pyarrow as pa

from carbonation.cli.utils import (
    make_passthru_parser,
    parse_data_to_ndarray,
    parse_data_to_pa_table,
)
from carbonation.measurand import Measurand, make_measurand


def grouper(iterable: Iterable, n: int):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)


RE_INT = re.compile(r"^\d+$")
RE_RANGE = re.compile(r"^\d+-\d*$")


def main() -> None:
    parser = argparse.ArgumentParser(prog="combine")
    parser.add_argument("-b", "--bits", type=int, default=8, help="input bits per word")
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

    measurands: List[Measurand] = []
    for spec in args.measurand:
        if RE_INT.match(spec):
            a = int(spec) - 1
            measurands.append(slice(a, a + 1))
        elif RE_RANGE.match(spec):
            a, b = spec.split("-")
            a = int(a) - 1
            if b == "":
                measurands.append(slice(a), None)
                continue

            b = int(b) - 1
            if a <= b:
                measurands.append(slice(a, b + 1))
            elif a > b:
                measurands.append(slice(a, b - 1, -1))
        else:
            if ";" not in spec:
                spec = spec.replace("/", ";")
            measurands.append(make_measurand(spec))

    print(measurands)
    line_parser = make_passthru_parser(args.passthru)

    # window = []
    for line in fileinput.input("-"):
        passthru, rest = line_parser(line)
        rest = rest.strip().split()
        print(passthru, end=" ")

        data = parse_data_to_ndarray([rest], dtype="u1")

        # data = parse_data_to_pa_table(rest, pa.uint8())
        # print(data)
        # data = pa.Table.from_arrays([data], names=[str(i) for i in range(len(data))])
        # data = np.array([int(x) for x in rest], dtype="u1")
        # data = pa.array([int(x) for x in rest], pa.uint8())

        for m in measurands:
            if isinstance(m, (int, slice)):
                for value in rest[m]:
                    print(value, end=" ")
            elif isinstance(m, Measurand):
                value = m.build(data)[0]
                print(value, end=" ")
            else:
                raise TypeError
            # print(value, end=" ")
        print()
