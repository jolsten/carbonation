import numpy as np
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from carbonation.measurand.interp import (
    IEEE16,
    IEEE32,
    IEEE64,
    TI32,
    TI40,
    Interp,
    InvalidInterpSize,
    MilStd1750A32,
    MilStd1750A48,
    OnesComplement,
    TwosComplement,
    Unsigned,
)
from carbonation.measurand.utils import _size_to_uint

from . import strategies as cst
from .conftest import ARRAY_SIZE


@given(cst.uint_and_size())
def test_unsigned(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = Unsigned().apply_ndarray(data, size)
    assert list(result) == [uint] * 10


@given(cst.uint_and_size())
def test_onescomp(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = OnesComplement().apply_ndarray(data, size)
    assert result.shape[0] == 10


@given(cst.uint_and_size())
def test_twoscomp(things):
    uint, size = things
    dtype = _size_to_uint(size)
    data = np.array([uint] * 10, dtype=dtype)
    result = TwosComplement().apply_ndarray(data, size)
    assert result.shape[0] == 10


@given(cst.uint(16))
def test_ieee16(uint):
    data = np.array([uint] * 10, dtype="u2")
    result = IEEE16().apply_ndarray(data, 16)
    assert result.shape[0] == 10


@given(cst.uint(32))
def test_ieee32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = IEEE32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(cst.uint(64))
def test_ieee64(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = IEEE64().apply_ndarray(data, 64)
    assert result.shape[0] == 10


@given(cst.uint(32))
def test_1750a32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = MilStd1750A32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(cst.uint(48))
def test_1750a48(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = MilStd1750A48().apply_ndarray(data, 48)
    assert result.shape[0] == 10


@given(cst.uint(32))
def test_ti32(uint):
    data = np.array([uint] * 10, dtype="u4")
    result = TI32().apply_ndarray(data, 32)
    assert result.shape[0] == 10


@given(cst.uint(40))
def test_ti40(uint):
    data = np.array([uint] * 10, dtype="u8")
    result = TI40().apply_ndarray(data, 40)
    assert result.shape[0] == 10


@pytest.mark.parametrize(
    "size, uint, int_1c, int_2c",
    [
        (3, 0b000, 0, 0),
        (3, 0b001, 1, 1),
        (3, 0b010, 2, 2),
        (3, 0b011, 3, 3),
        (3, 0b100, -3, -4),
        (3, 0b101, -2, -3),
        (3, 0b110, -1, -2),
        (3, 0b111, 0, -1),
        (8, 0b00000000, 0, 0),
        (8, 0b00000001, 1, 1),
        (8, 0b00000010, 2, 2),
        (8, 0b01111110, 126, 126),
        (8, 0b01111111, 127, 127),
        (8, 0b10000000, -127, -128),
        (8, 0b10000001, -126, -127),
        (8, 0b11111101, -2, -3),
        (8, 0b11111110, -1, -2),
        (8, 0b11111111, 0, -1),
    ],
)
class TestSignedIntegers:
    def test_1c_3bit_ndarray(self, size, uint, int_1c, int_2c):
        strategy = OnesComplement()
        data = np.array([uint] * ARRAY_SIZE, dtype="u1")
        results = strategy.apply_ndarray(data, size)
        assert list(results) == [int_1c] * ARRAY_SIZE

    def test_1c_3bit_paarray(self, size, uint, int_1c, int_2c):
        strategy = OnesComplement()
        data = pa.array(np.array([uint] * ARRAY_SIZE, dtype="u1"))
        results = strategy.apply_paarray(data, size)
        assert results.to_pylist() == [int_1c] * ARRAY_SIZE

    def test_2c_3bit_ndarray(self, size, uint, int_1c, int_2c):
        strategy = TwosComplement()
        data = np.array([uint] * ARRAY_SIZE, dtype="u1")
        assert list(strategy.apply_ndarray(data, size)) == [int_2c] * ARRAY_SIZE

    def test_2c_3bit_paarray(self, size, uint, int_1c, int_2c):
        strategy = TwosComplement()
        data = pa.array(np.array([uint] * ARRAY_SIZE, dtype="u1"))
        assert strategy.apply_paarray(data, size).to_pylist() == [int_2c] * ARRAY_SIZE


class FloatInterpTester:
    strategy: Interp = ...
    size: int = ...
    dtype: str = ...

    def test_ndarray(self, uint, value):
        data = np.array([uint] * ARRAY_SIZE, dtype=self.dtype)
        result = self.strategy.apply_ndarray(data, self.size)
        assert list(result) == pytest.approx([value] * ARRAY_SIZE)

    def test_paarray(self, uint, value):
        data = pa.array(np.array([uint] * ARRAY_SIZE, dtype=self.dtype))
        result = self.strategy.apply_paarray(data, self.size)
        assert result.to_pylist() == pytest.approx([value] * ARRAY_SIZE)


@pytest.mark.parametrize(
    "uint, value",
    [
        (0x7FFFFF7F, 0.9999998 * 2**127),
        (0x4000007F, 0.5 * 2**127),
        (0x50000004, 0.625 * 2**4),
        (0x40000001, 0.5 * 2**1),
        (0x40000000, 0.5 * 2**0),
        (0x400000FF, 0.5 * 2**-1),
        (0x40000080, 0.5 * 2**-128),
        (0x00000000, 0.0 * 2**0),
        (0x80000000, -1.0 * 2**0),
        (0xBFFFFF80, -0.5000001 * 2**-128),
        (0x9FFFFF04, -0.7500001 * 2**4),
    ],
)
class TestMilStd1750a32(FloatInterpTester):
    strategy = MilStd1750A32()
    size = 32
    dtype = "u4"


@pytest.mark.parametrize(
    "uint, value",
    [
        (0x4000007F0000, 0.5 * 2**127),
        (0x400000000000, 0.5 * 2**0),
        (0x400000FF0000, 0.5 * 2**-1),
        (0x400000800000, 0.5 * 2**-128),
        (0x8000007F0000, -1.0 * 2**127),
        (0x800000000000, -1.0 * 2**0),
        (0x800000FF0000, -1.0 * 2**-1),
        (0x800000800000, -1.0 * 2**-128),
        (0x000000000000, 0.0 * 2**0),
        (0xA00000FF0000, -0.75 * 2**-1),
    ],
)
class TestMilStd1750a48(FloatInterpTester):
    strategy = MilStd1750A48()
    size = 48
    dtype = "u8"


###################
# Exception Tests #
###################


@given(st.integers())
def test_invalid_size_ieee32(size):
    assume(size != 32)
    with pytest.raises(InvalidInterpSize):
        IEEE32().apply_ndarray(np.array([0], dtype="u4"), size)
