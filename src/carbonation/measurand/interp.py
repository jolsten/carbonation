from typing import ClassVar, Optional

import numpy as np
import pyarrow as pa
import typeconvert.ufunc as tcu

from carbonation.measurand.generic import MeasurandModifier, ObjectFactory


class Interp(MeasurandModifier):
    SIZE: ClassVar[Optional[int]] = None

    def apply_paarray(self, data: pa.Array, bits: int) -> pa.Array:
        result = self.apply_ndarray(data.to_numpy(), bits)
        return pa.array(result)


class InvalidInterpType(ValueError):
    def __init__(self, interp_spec: str) -> None:
        self.msg = f'"{interp_spec}" is not a valid interpretation specification.'

    def __str__(self) -> str:
        return self.msg


class InvalidInterpSize(ValueError):
    def __init__(self, cls: Interp, expected_size: int, received_size: int) -> None:
        self.msg = f"{cls.__name__} expects a word with size {expected_size}"
        f"but was provided a word with size {received_size}"

    def __str__(self) -> str:
        return self.msg


interp = ObjectFactory()


@interp.register("u")
class Unsigned(Interp):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return data


@interp.register("1c")
class OnesComplement(Interp):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return tcu.onescomp(data, bits)


@interp.register("2c")
class TwosComplement(Interp):
    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        return tcu.twoscomp(data, bits)


@interp.register("ieee16")
class IEEE16(Interp):
    SIZE: ClassVar[Optional[int]] = 16

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f2")


@interp.register("ieee32")
class IEEE32(Interp):
    SIZE: ClassVar[Optional[int]] = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f4")


@interp.register("ieee64")
class IEEE64(Interp):
    SIZE: ClassVar[Optional[int]] = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return data.view(">f8")


@interp.register("1750a32")
class MilStd1750A32(Interp):
    SIZE: ClassVar[Optional[int]] = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.milstd1750a32(data)


@interp.register("1750a48")
class MilStd1750A48(Interp):
    SIZE: ClassVar[Optional[int]] = 48

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.milstd1750a48(data)


@interp.register("ti32")
class TI32(Interp):
    SIZE: ClassVar[Optional[int]] = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.ti32(data)


@interp.register("ti40")
class TI40(Interp):
    SIZE: ClassVar[Optional[int]] = 40

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.ti40(data)


@interp.register("ibm32")
class IBM32(Interp):
    SIZE: ClassVar[Optional[int]] = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.ibm32(data)


@interp.register("ibm64")
class IBM64(Interp):
    SIZE: ClassVar[Optional[int]] = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.ibm64(data)


@interp.register("dec32")
class DEC32(Interp):
    SIZE: ClassVar[Optional[int]] = 32

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.dec32(data)


@interp.register("dec64")
class DEC64(Interp):
    SIZE: ClassVar[Optional[int]] = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.dec64(data)


@interp.register("dec64g")
class DEC64G(Interp):
    SIZE: ClassVar[Optional[int]] = 64

    def apply_ndarray(self, data: np.ndarray, bits: int) -> np.ndarray:
        if bits != self.SIZE:
            raise InvalidInterpSize(self.__class__, self.SIZE, bits)
        return tcu.dec64g(data)


def make_interp(spec: str) -> Interp:
    return interp.create(spec)
