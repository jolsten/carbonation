import numpy as np
import pyarrow as pa

from carbonation.measurand.utils import _size_to_uint

ARRAY_SIZE = 100


def _make_sample_data(word_size: int) -> np.ndarray:
    dtype = _size_to_uint(word_size)
    stop = 2**word_size + 1
    one = np.arange(start=1, stop=stop, dtype=np.uint16)
    one = np.fmod(one, np.uint16(2**word_size))
    return np.repeat(np.atleast_2d(one), ARRAY_SIZE, axis=0).astype(dtype)


SAMPLE_NDARRAY = {word_size: _make_sample_data(word_size) for word_size in [8, 10, 12]}


def numpy_2d_array_to_arrow_table(array: np.ndarray) -> pa.Table:
    arrays = [pa.array(col) for col in array.T]
    table = pa.Table.from_arrays(arrays, names=[str(i) for i in range(len(arrays))])
    return table


SAMPLE_PAARRAY = {
    word_size: numpy_2d_array_to_arrow_table(SAMPLE_NDARRAY[word_size])
    for word_size in [8, 10, 12]
}
