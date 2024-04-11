from typing import List

import pytest
from carbonation.utils import expand_column_list


@pytest.mark.parametrize(
    "spec, list_",
    [
        ("1", [1]),
        ("1-2", [1, 2]),
        ("1,2", [1, 2]),
        ("1-2,3,4", [1, 2, 3, 4]),
        ("1++1<4", [1, 2, 3, 4]),
        ("1++2<4", [1, 3]),
        ("1++4<16", [1, 5, 9, 13]),
        ("1,2,1++4<16", [1, 2, 1, 5, 9, 13]),
    ],
)
def test_expand_column_list(spec: str, list_: List[int]) -> None:
    assert expand_column_list(spec) == list_
