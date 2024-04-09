import re
from functools import cached_property
from typing import Callable, Optional, Union

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BeforeValidator

from carbonation.measurand.generic import MeasurandModifier

RE_SCALEFACTOR = re.compile(
    r"^(?:EUC)?\[?(?:(?P<data_bias>\S+?),)?(?P<scale_factor>\S+?)(?:,(?P<scaled_bias>\S+?))?\]?$",
    re.IGNORECASE,
)
RE_FLOAT = re.compile(r"[\d\.\+\-\*\/]+")


def _validate_float(spec: str) -> float:
    try:
        return np.float32(float(spec))
    except ValueError:
        # Power: Replace ^ (bitwise and) with ** (raise to power)
        spec = spec.replace("^", "**")
        if RE_FLOAT.match(spec):
            return np.float32(eval(spec))
    raise ValueError


Float = Annotated[float, BeforeValidator(_validate_float)]


class EUC(MeasurandModifier):
    pass


class ScaleFactorEUC(EUC):
    """The Engineering Unit Conversion to apply against the `Parameter`.

    `EUC` parses a `str` containing the specification for a desired
    engineering unit conversion. It also provides a method to apply the
    scale factor to a `numpy.ndarray` vector and return a scaled vector.

    The scale factor is defined by three components: data_bias, scale_factor,
    and scaled_bias. The scale factor is applied according to the following
    expression:

        (PV + data_bias) * scale_factor + scaled_bias

    Parameters
    ----------
    spec : str
        The `Component` specification string.

    one_based : bool, default True
        Flag indicating whether the numbers shall be treated as 1-based
        or 0-based.

    word_size : int, default 8
        The size of the underlying word

    Attributes
    ----------
    data_bias : float or None
    scale_factor : float or None
    scaled_bias: float or None
    """

    data_bias: Optional[Float] = None
    scale_factor: Optional[Float] = None
    scaled_bias: Optional[Float] = None

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        dtype = np.dtype(">f8")

        # If the data is not already floating, and the EUC is not a No-Op:
        # Convert the data to float
        if not np.issubdtype(data.dtype, np.floating) and any(
            v is not None for v in (self.data_bias, self.scale_factor, self.scaled_bias)
        ):
            data = data.astype(dtype)

        if self.data_bias is not None:
            data += self.data_bias

        if self.scale_factor is not None:
            data = data * self.scale_factor

        if self.scaled_bias is not None:
            data += self.scaled_bias

        return data

    def apply_paarray(self, data: pa.Array, bits: int) -> pa.Array:
        dtype = pa.from_numpy_dtype(np.dtype(">f8"))

        # If the data is not already floating, and the EUC is not a No-Op:
        # Convert the data to float
        if not pa.types.is_floating(data.type) and any(
            v is not None for v in (self.data_bias, self.scale_factor, self.scaled_bias)
        ):
            data = data.cast(dtype)

        if self.data_bias is not None:
            data = pc.add(data, pa.scalar(self.data_bias, type=dtype))

        if self.scale_factor is not None:
            data = pc.multiply(data, pa.scalar(self.scale_factor, type=dtype))

        if self.scaled_bias is not None:
            data = pc.add(data, pa.scalar(self.scaled_bias, type=dtype))

        return data


class FunctionEUC(EUC):
    func: Callable[[Union[int, float]], float]

    @cached_property
    def vectorized_func(self) -> Callable:
        return np.vectorize(self.func)

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        data = data.astype(np.float64)
        return self.vectorized_func(data)


def make_euc(spec: str) -> EUC:
    if match := RE_SCALEFACTOR.match(spec):
        return ScaleFactorEUC(
            data_bias=match.group("data_bias"),
            scale_factor=match.group("scale_factor"),
            scaled_bias=match.group("scaled_bias"),
        )
    elif "PV" in spec.upper():
        raise NotImplementedError
    raise ValueError(f"EUC spec {spec!r} not valid")
