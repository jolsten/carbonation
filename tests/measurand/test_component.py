import hypothesis.strategies as st
import pytest
from hypothesis import given

from carbonation.measurand.component import _RE_COMPONENT, make_component

from . import strategies as cst
from .cases import Example, component_test_cases
from .conftest import ARRAY_SIZE, SAMPLE_NDARRAY, SAMPLE_PAARRAY


@given(
    cst.component_spec(),
)
def test_re_component(spec):
    assert _RE_COMPONENT.match(spec)


@given(
    st.integers(min_value=1, max_value=1023),
    st.tuples(
        st.integers(min_value=0, max_value=7),
        st.integers(min_value=0, max_value=7),
    ),
    st.booleans(),
)
def test_re_component_with_bits(word, bits, reversed):
    a, b = bits
    if a == b:
        bits = a
    else:
        bits = f"{a}-{b}"
    spec2 = f'{word}:{bits}{"R" if reversed else ""}'
    print(spec2)
    assert _RE_COMPONENT.match(spec2)


@given(cst.word(one_based=True))
def test_component_1_based_byte(word):
    spec = f"{word}"
    c = make_component(spec)
    assert c.word == word - 1
    assert c.mask is None
    assert c.shift == 0


@given(cst.word(one_based=False))
def test_component_0_based_byte(word):
    spec = f"{word}"
    c = make_component(spec, one_based=False)
    assert c.word == word
    assert c.mask is None
    assert c.shift == 0


@given(
    cst.word(one_based=True),
    cst.bit(one_based=True, word_size=16),
)
def test_component_1_based_bit(word, bit):
    spec = f"{word}:{bit}"
    print(spec)
    c = make_component(spec, one_based=True)
    print("c.word =", c.word)
    print("c.mask =", c.mask)
    assert c.word == word - 1
    assert c.mask == 2 ** (bit - 1)


@given(
    cst.word(one_based=False),
    cst.bit(one_based=False, word_size=16),
)
def test_component_0_based_bit(word, bit):
    spec = f"{word}:{bit}"
    c = make_component(spec, one_based=False)
    assert c.word == word
    assert c.mask == 2**bit


@pytest.mark.parametrize(
    "spec, word_size, size",
    [
        ("1", 8, 8),
        ("1", 10, 10),
        ("1:1-4", 8, 4),
        ("1:1-2", 8, 2),
    ],
)
def test_component_size(spec, word_size, size):
    c = make_component(spec, word_size=word_size)
    assert c.size == size


@pytest.mark.parametrize(
    "spec",
    [
        "1:not-valid",
        "1=2",
        "a",
    ],
)
def test_component_invalid_spec(spec):
    with pytest.raises(ValueError):
        make_component(spec)


@pytest.mark.parametrize("case", component_test_cases)
class TestBuildComponent:
    def test_build_ndarray(self, case: Example):
        c = make_component(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        out = c._build_ndarray(SAMPLE_NDARRAY[case.word_size])
        print(repr(out[0:2]))
        assert list(out) == list([case.result] * ARRAY_SIZE)

    def test_build_paarray(self, case: Example):
        c = make_component(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        out = c._build_paarray(SAMPLE_PAARRAY[case.word_size])
        assert out.to_pylist() == list([case.result] * ARRAY_SIZE)


@given(cst.word_and_word_size())
def test_parameter(word_and_word_size):
    word, word_size = word_and_word_size
    p = make_component(f"{word}", word_size=word_size)
    assert list(p._build_ndarray(SAMPLE_NDARRAY[word_size])) == list(
        [word % 2**word_size] * ARRAY_SIZE
    )
