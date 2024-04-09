from typing import List

from hypothesis import strategies as st

from carbonation.measurand.interp import interp

MAX_FRAME_SIZE = 4096
MAX_PARAMETER_SIZE = 64

valid_parameter_sizes = st.integers(min_value=1, max_value=64)
bit_positions = st.integers(min_value=0, max_value=15)


@st.composite
def euc_float(draw):
    return draw(st.floats(min_value=1e-12, max_value=1e12))


@st.composite
def uint(draw, size: int):
    return draw(st.integers(min_value=0, max_value=2**size - 1))


@st.composite
def uint_and_size(draw, max_size: int = 64):
    size = draw(st.integers(min_value=1, max_value=max_size))
    value = draw(uint(size))
    return value, size


@st.composite
def word(draw, one_based: bool = True, word_size: int = 8):
    min = 1 if one_based else 0
    max = 2**word_size if one_based else 2**word_size - 1
    return draw(st.integers(min_value=min, max_value=max))


@st.composite
def word_and_word_size(draw, word_sizes: List[int] = [8, 10, 12]):
    word_size = draw(st.sampled_from(word_sizes))
    word_ = draw(word(word_size=word_size))
    return word_, word_size


@st.composite
def bit(draw, one_based: bool = True, word_size: int = 8) -> int:
    return draw(
        st.integers(
            min_value=1 if one_based else 0,
            max_value=word_size if one_based else word_size - 1,
        )
    )


@st.composite
def component_spec(draw, one_based: bool = True, word_size: int = 8) -> str:
    word_ = draw(word(one_based=one_based))
    msb = draw(bit(one_based=one_based, word_size=word_size))
    lsb = draw(bit(one_based=one_based, word_size=word_size))
    reverse = "R" if draw(st.booleans()) else ""
    if msb == lsb:
        return f"{word_}:{lsb}{reverse}"
    elif msb < lsb:
        msb, lsb = lsb, msb
    return f"{word_}:{lsb}-{msb}{reverse}"


@st.composite
def parameter_spec(
    draw, max_size: int = 64, one_based: bool = True, word_size: int = 8
) -> str:
    max_components = max_size // word_size
    components = draw(
        st.lists(
            component_spec(one_based=one_based, word_size=word_size),
            min_size=1,
            max_size=max_components,
        )
    )
    return "[" + "+".join(components) + "]"


@st.composite
def interp_spec(draw):
    return draw(st.sampled_from(interp.registry.keys()))


@st.composite
def euc_scalefactor_spec(draw):
    parts = draw(st.lists(euc_float(), min_size=1, max_size=3))
    return f"EUC[{','.join(parts)}]"
