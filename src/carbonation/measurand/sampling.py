from typing import Literal

import numpy as np
import pyarrow as pa

from carbonation.measurand.generic import MeasurandModifier

SamplingStrategy = Literal["mean", "mode", "max", "min"]


class Sampling(MeasurandModifier):
    window: int
    mode: SamplingStrategy

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        raise NotImplementedError

    def apply_paarray(self, data: pa.Table, bits: int) -> pa.Table:
        raise NotImplementedError
