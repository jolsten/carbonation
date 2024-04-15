import io
import sys

import pytest
from carbonation.cli.measurand import main

from tests.measurand.cases import Example, measurand_test_cases

DATE = "2020-01-02T03:04:05"


def make_data(bits: int) -> str:
    size = 2**bits
    return f"{DATE} " + " ".join([str((x + 1) % size) for x in range(size)])


DATA_8BIT = make_data(8)
DATA_10BIT = make_data(10)

modified_test_cases = []
for case in measurand_test_cases:
    if "-" in case.spec:
        pass
    elif not case.one_based:
        pass
    else:
        modified_test_cases.append(case)


@pytest.mark.parametrize(
    "case",
    modified_test_cases,
)
def test_measurand_one_by_one(capsys, monkeypatch, case: Example):
    with monkeypatch.context() as m:
        if case.word_size == 8:
            in_ = io.StringIO(DATA_8BIT)
        else:
            in_ = io.StringIO(make_data(case.word_size))
        m.setattr(sys, "stdin", in_)

        argv = [sys.argv[0], case.spec]
        if case.word_size != 8:
            argv.append(f"-b={case.word_size}")
        m.setattr(sys, "argv", argv)

        main()
        out, err = capsys.readouterr()
        print("argv =", argv)
        print("out =", out)
        print("result =", case.result)

        out = out.rstrip()
        time, *rest = out.strip().split()
        out = [float(x) for x in rest]
        assert time == DATE
        assert out == pytest.approx([case.result])
        assert err == ""
