import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from carbonation.measurand.component import make_component
from carbonation.measurand.parameter import make_parameter

from . import strategies as cst
from .cases import Example, component_test_cases, parameter_test_cases
from .conftest import ARRAY_SIZE, SAMPLE_NDARRAY, SAMPLE_PAARRAY


@given(st.lists(st.sampled_from(component_test_cases), min_size=1, max_size=8))
def test_parameter_components(cases):
    component_specs = [case.spec for case in cases]
    comps = (make_component(c) for c in component_specs)
    spec = "+".join(component_specs)
    param = make_parameter(spec)
    assume(param.size <= 64)
    print(comps, param.components)
    assert list(param.components) == list(comps)


@pytest.mark.parametrize(
    "spec, word_size, size",
    [
        ("1", 8, 8),
        ("1", 10, 10),
        ("1+2", 8, 16),
        ("1:1-4+2:5-8", 8, 8),
        ("1-4", 8, 32),
        ("1-3", 10, 30),
    ],
)
def test_parameter_size(spec, word_size, size):
    r = make_parameter(spec, word_size=word_size)
    assert r.size == size


@given(cst.parameter_spec())
def test_parameter_eq(spec):
    print("spec =", spec)
    p1 = make_parameter(spec)
    p2 = make_parameter(spec)
    print("param 1:", p1)
    print("param 2:", p2)
    assert p1 == p2


@given(cst.parameter_spec(), cst.parameter_spec())
def test_parameter_ne(spec1, spec2):
    assume(spec1 != spec2)
    print(spec1, spec2)
    p1 = make_parameter(spec1)
    p2 = make_parameter(spec2)
    print(p1, p2, p1 == p2, p1 != p2)
    assert p1 != p2


@pytest.mark.parametrize("case", parameter_test_cases)
class TestBuildParameter:
    def test_build_ndarray(self, case: Example):
        p = make_parameter(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        print("spec =", case.spec, "result =", f"{case.result:0{p.size}b}")
        out = p._build_ndarray(SAMPLE_NDARRAY[case.word_size])
        assert list(out) == list([case.result] * ARRAY_SIZE)

    def test_build_paarray(self, case: Example):
        p = make_parameter(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        print("spec =", case.spec, "result =", f"{case.result:0{p.size}b}")
        out = p._build_paarray(SAMPLE_PAARRAY[case.word_size])
        assert out.to_pylist() == list([case.result] * ARRAY_SIZE)

    def test_build(self, case: Example):
        p = make_parameter(
            case.spec, word_size=case.word_size, one_based=case.one_based
        )
        print("spec =", case.spec, "result =", f"{case.result:0{p.size}b}")
        out = p.build(SAMPLE_NDARRAY[case.word_size])
        assert list(out) == list([case.result] * ARRAY_SIZE)

        out = p.build(SAMPLE_PAARRAY[case.word_size])
        assert out.to_pylist() == list([case.result] * ARRAY_SIZE)
