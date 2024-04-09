import numpy as np
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from carbonation.measurand.utils import (
    _bit_range_to_mask_and_shift,
    _expand_component_range,
    _expand_list,
    _range_to_tuple,
    _reverse_bits_ndarray,
    _reverse_bits_paarray,
    _size_to_uint,
)

from .conftest import ARRAY_SIZE


@pytest.mark.parametrize(
    "spec,result",
    [
        ("1-4", "1+2+3+4"),
        ("4-1", "4+3+2+1"),
        ("1-2R", "1R+2R"),
        ("1-4:2-9", "1:2-9+2:2-9+3:2-9+4:2-9"),
    ],
)
def test_expand_component_range(spec, result):
    assert _expand_component_range(spec) == result


@given(st.integers(min_value=0, max_value=16), st.integers(min_value=0, max_value=16))
def test_range_to_tuple(a, b):
    assume(a < b)
    assert _range_to_tuple(f"{a}-{b}") == (a, b)
    assert _range_to_tuple(f"{b}-{a}") == (a, b)


@given(st.integers(min_value=0, max_value=1024))
def test_expand_list_single_digit(value):
    assert _expand_list(f"{value}") == [value]


@pytest.mark.parametrize(
    "spec, result",
    [
        ("1-8", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("0-7", [0, 1, 2, 3, 4, 5, 6, 7]),
        ("8-1", [8, 7, 6, 5, 4, 3, 2, 1]),
        ("7-0", [7, 6, 5, 4, 3, 2, 1, 0]),
    ],
)
def test_expand_list(spec, result):
    assert _expand_list(spec) == result


def test_expand_list_typeerror_1():
    with pytest.raises(TypeError):
        _expand_list(1)


def test_expand_list_valueerror():
    with pytest.raises(ValueError):
        _expand_list("1.23")


def test_expand_list_typeerror_2():
    with pytest.raises(ValueError):
        _expand_list("1:1-n")


@pytest.mark.parametrize(
    "spec, shift, mask",
    [
        ("1", 0, 1),
        ("2", 1, 2),
        ("3", 2, 4),
        ("4", 3, 8),
        ("5", 4, 16),
        ("6", 5, 32),
        ("7", 6, 64),
        ("8", 7, 128),
    ],
)
def test_bit_spec_to_mask_and_rshift_digit(spec, mask, shift):
    one_based = True
    a, b = int(spec), int(spec)
    if one_based:
        a, b = a - 1, b - 1
    assert _bit_range_to_mask_and_shift(a, b) == (mask, shift)


@pytest.mark.parametrize(
    "spec, one_based, mask, shift",
    [
        ("1-2", True, 0b00000011, 0),
        ("2-1", True, 0b00000011, 0),
        ("1-3", True, 0b00000111, 0),
        ("3-1", True, 0b00000111, 0),
        ("1-4", True, 0b00001111, 0),
        ("4-1", True, 0b00001111, 0),
        ("5-8", True, 0b11110000, 4),
        ("8-5", True, 0b11110000, 4),
        ("7-8", True, 0b11000000, 6),
        ("8-7", True, 0b11000000, 6),
    ],
)
def test_bit_spec_to_mask_and_rshift_range(spec, one_based, mask, shift):
    a, b = [int(x) for x in spec.split("-")]
    if one_based:
        a, b = a - 1, b - 1
    assert _bit_range_to_mask_and_shift(a, b) == (mask, shift)


@pytest.mark.parametrize(
    "a, b",
    [
        (1, "a"),
    ],
)
def test_bit_spec_to_mask_and_rshift_exception(a, b):
    with pytest.raises(TypeError):
        _bit_range_to_mask_and_shift(a, b)


@given(st.integers(min_value=1, max_value=64))
def test_size_to_uint(size):
    dtype = np.dtype(_size_to_uint(size))
    assert size / (dtype.itemsize * 8) <= 1.0


@given(st.integers(min_value=65, max_value=128))
def test_size_to_uint_too_big(size):
    with pytest.raises(ValueError):
        _size_to_uint(size)


@pytest.mark.parametrize(
    "input, output, dtype, word_size",
    [
        (0x00, 0x00, "u1", 8),
        (0x0F, 0xF0, "u1", 8),
        (0xF0, 0x0F, "u1", 8),
        (0xFF, 0xFF, "u1", 8),
        (0x000, 0x000, "u2", 12),
        (0x00F, 0xF00, "u2", 12),
        (0xF00, 0x00F, "u2", 12),
        (0xFFF, 0xFFF, "u2", 12),
        (0x0000, 0x0000, "u2", 16),
        (0x000F, 0xF000, "u2", 16),
        (0xF000, 0x000F, "u2", 16),
        (0xFFFF, 0xFFFF, "u2", 16),
        (0x55, 0xAA, "u1", 8),
        (0xAA, 0x55, "u1", 8),
        (0x555, 0xAAA, "u2", 12),
        (0xAAA, 0x555, "u2", 12),
        (0x5555, 0xAAAA, "u2", 16),
        (0xAAAA, 0x5555, "u2", 16),
    ],
)
class TestReverseBits:
    def test_ndarray(self, input, output, dtype, word_size):
        data = np.array([input] * ARRAY_SIZE, dtype=dtype)
        assert list(_reverse_bits_ndarray(data, word_size)) == list(
            [output] * ARRAY_SIZE
        )

    def test_paarray(self, input, output, dtype, word_size):
        data = pa.array([input] * ARRAY_SIZE, type=dtype)
        assert _reverse_bits_paarray(data, word_size).to_pylist() == list(
            [output] * ARRAY_SIZE
        )
