import collections
import multiprocessing
from typing import Tuple, TypeVar

import numpy as np
import numpy.typing as npt


SAHType = TypeVar("SAHType", bound="SharedArrayHelper")


SharedArrayMeta = collections.namedtuple(
    "SharedArrayMeta", ["buffer", "dtype", "shape"])


class SharedArrayHelper(object):

    def __init__(
        self,
        data: npt.ArrayLike
    ) -> None:
        self._data = np.asarray(data)
        # create a shared array as buffer for the data
        self.buffer = multiprocessing.RawArray(self.cdtype, data.size)
        # replace the referenced data set by a new numpy view of the buffer
        self._data = self.get_data()
        # set the array values
        np.copyto(self._data, data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        return self._data.dtype

    @property
    def cdtype(self):
        return np.ctypeslib.as_ctypes_type(self.dtype)

    def get_meta(self) -> SharedArrayMeta:
        return SharedArrayMeta(self.buffer, self.dtype, self.shape)

    @classmethod
    def from_meta(
        cls,
        metadata: SharedArrayMeta
    ) -> SAHType:
        new = cls.__new__(cls)
        new.buffer = metadata.buffer  # this should already contain the data
        new._data = np.frombuffer(  # array view of the data
            new.buffer, dtype=metadata.dtype).reshape(metadata.shape)
        return new

    def get_data(self) -> npt.NDArray:
        return np.frombuffer(self.buffer, dtype=self.dtype).reshape(self.shape)
