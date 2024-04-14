import io
import sys

import pytest
from carbonation.cli.reframe import main


@pytest.mark.parametrize(
    "input_, reframe_args, output",
    [
        (
            "2020-01-02T03:04:05 1023 510 511 512 513\n",
            ("1-5:2-9", "--bits=10"),
            "2020-01-02T03:04:05  255  255  255    0    0",
        ),
        (
            "2020-01-02T03:04:05 1023 510 511 512 513\n",
            ("1++1<5:2-9", "--bits=10"),
            "2020-01-02T03:04:05  255  255  255    0    0",
        ),
        (
            "123456789 2020-01-02T03:04:05 1 2 3 4 5 6",
            ("2-5", "--passthru=2"),
            "123456789 2020-01-02T03:04:05   2   3   4   5",
        ),
        (
            "1 2 3 4 5 6",
            ("2-5", "--passthru=0"),
            "   2   3   4   5",
        ),
    ],
)
def test_reframe(capsys, monkeypatch, input_: str, reframe_args: str, output: str):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdin", io.StringIO(input_))
        argv = [sys.argv[0]]
        argv.extend(reframe_args)
        m.setattr(sys, "argv", argv)
        main()
        out, err = capsys.readouterr()
        out = out.rstrip()
        assert out == output
        assert err == ""


def test_leading_whitespace(capsys, monkeypatch):
    input_ = " 123456789 2020-01-02T03:04:05 1 2 3 4 5 6"
    reframe_args = ("2-5", "--passthru=2")
    output = "123456789 2020-01-02T03:04:05   2   3   4   5"

    with monkeypatch.context() as m:
        m.setattr(sys, "stdin", io.StringIO(input_))
        argv = [sys.argv[0]]
        argv.extend(reframe_args)
        m.setattr(sys, "argv", argv)
        main()
        out, err = capsys.readouterr()
        out = out.rstrip()
        assert out == output
        assert err == ""
