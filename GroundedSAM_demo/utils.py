import numpy as np

from typing_extensions import Any, List, Optional

class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic functionality for appending and final concatenation.
    """

    def __init__(self, batch_size: int, data: Optional[List] = None) -> None:
        self.batch_size = batch_size

        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self) -> int:
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx) -> Any:
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size:(idx + 1) * self.batch_size]

    def cat(self, data: Any, dim: int = 0) -> None:
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data: Any) -> None:
        self.data.append(data)

    def stack(self, dim: int = 0) -> None:
        self.data = torch.stack(self.data, dim=dim)

