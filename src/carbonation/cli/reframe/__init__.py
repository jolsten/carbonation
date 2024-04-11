import argparse
import fileinput
import re

import numpy as np

from carbonation.cli.utils import make_passthru_parser
from carbonation.measurand.utils import size_to_uint
from carbonation.utils import expand_column_list

COL_RANGE = re.compile(r"^\d+-\d+$")
BIT_RANGE = re.compile(r"^(?P<column_range>\d+-\d+)\:(?P<bit_range>\S+)$")


class ReframeSpec:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.col_list = []
        self.start_bit = None
        self.stop_bit = None
        if ":" in spec:
            col_range, bit_range = spec.split(":")
            start_bit, stop_bit = bit_range.split("-")
            self.start_bit = int(start_bit)
            self.stop_bit = int(stop_bit)
            spec = col_range
        self.col_list = expand_column_list(spec)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(spec={self.spec!r})"


def main():
    parser = argparse.ArgumentParser(prog="reframe")
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
    parser.add_argument("reframe_spec", type=str, help="reframe spec")
    args = parser.parse_args()

    width = len(f"{2**args.bits-1}")
    reframe = ReframeSpec(args.reframe_spec)
    col_list = [x - 1 for x in reframe.col_list]

    line_parser = make_passthru_parser(args.passthru)
    dtype = size_to_uint(args.bits)

    for line in fileinput.input("-"):
        passthru, rest = line_parser(line)
        rest = rest.strip().split()
        print(passthru, end=" ")

        data = np.array([int(x) for x in rest], dtype=dtype)

        if reframe.stop_bit is not None:
            data = np.bitwise_and(data, 2**reframe.stop_bit - 1)
        if reframe.start_bit is not None:
            data = np.right_shift(data, reframe.start_bit - 1)

        for value in data[col_list]:
            print(f"{value:>{width}}", end=" ")
