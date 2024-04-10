import argparse
import datetime
import fileinput
from itertools import zip_longest
from typing import Iterable, List, Literal, Tuple

import numpy as np
from dateutil.parser import parse as parse_datetime

from carbonation.measurand import Measurand, make_measurand


def grouper(iterable: Iterable, n: int):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=None)


def parse_line(
    line: str, dtype: Literal["u1", "u2", "u4", "u8"]
) -> Tuple[datetime.datetime, np.ndarray]:
    time, *values = line.strip().split()
    time = parse_datetime(time)
    values = np.array([int(x) for x in values], dtype=dtype)
    return time, values


def main() -> None:
    parser = argparse.ArgumentParser(prog="combine")
    parser.add_argument("-b", "--bits", type=int, default=8, help="input bits per word")
    parser.add_argument(
        "-c", "--chunk-size", type=int, default=1000, help="processing chunk size"
    )
    parser.add_argument("measurand", type=str, nargs="+", help="measurand definition")
    args = parser.parse_args()

    measurands: List[Measurand] = []
    for spec in args.measurand:
        spec = spec.replace(";", ";")
        measurands.append(make_measurand(spec))

    window = []
    for line in fileinput.input("-"):
        time, data = parse_line(line, dtype="u1")
        print(time, end=" ")

        for m in measurands:
            value = m.build(data)[0]
            print(value, end=" ")
