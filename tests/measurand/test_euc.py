import numpy as np
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from carbonation.measurand.euc import RE_SCALEFACTOR, ScaleFactorEUC, make_euc

from . import strategies as cst
from .conftest import ARRAY_SIZE


@given(st.lists(cst.euc_float(), min_size=1, max_size=3))
def test_scale_factor_regex(args):
    csv = ",".join([str(x) for x in args])
    for spec in [f"EUC[{csv}]", f"euc[{csv}]", f"[{csv}]", csv]:
        assert RE_SCALEFACTOR.match(spec)


@given(
    cst.euc_float(),
    cst.euc_float(),
    cst.euc_float(),
)
def test_make_euc_scalefactor(db, sf, sb):
    for spec, result in {
        f"EUC[{sf}]": ScaleFactorEUC(scale_factor=sf),
        f"EUC[{db},{sf}]": ScaleFactorEUC(data_bias=db, scale_factor=sf),
        f"EUC[{db},{sf},{sb}]": ScaleFactorEUC(
            data_bias=db, scale_factor=sf, scaled_bias=sb
        ),
    }.items():
        euc = make_euc(spec)
        assert euc == result


@given(
    st.integers(min_value=0, max_value=255),
    cst.euc_float(),
    cst.euc_float(),
    cst.euc_float(),
)
def test_euc_apply_ndarray(val: int, db: float, sf: float, sb: float):
    euc = ScaleFactorEUC(data_bias=db, scale_factor=sf, scaled_bias=sb)
    data = np.array([val] * ARRAY_SIZE, dtype="u1")
    result = euc.apply_ndarray(data, 8)
    answer = (data.astype("f8") + db) * sf + sb
    assert result.tolist() == pytest.approx(answer.tolist())


@given(
    st.integers(min_value=0, max_value=255),
    cst.euc_float(),
    cst.euc_float(),
    cst.euc_float(),
)
def test_euc_apply_paarray(val: int, db: float, sf: float, sb: float):
    euc = ScaleFactorEUC(data_bias=db, scale_factor=sf, scaled_bias=sb)
    data = pa.array(np.array([val] * ARRAY_SIZE, dtype="u1"))
    result = euc.apply_paarray(data, 8)
    answer = (float(val) + db) * sf + sb
    assert result.to_pylist() == pytest.approx([answer] * ARRAY_SIZE)


def test_scale_factor_invalid():
    with pytest.raises(ValueError):
        make_euc("not a valid scale factor")
