"""Type-hints for held attributes not intended for serialization."""

from typing import Annotated, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

# Pydantic type-hint to exclude tensor from JSON schema/model dump methods
ExcludedTensor = SkipJsonSchema[
    Annotated[Optional[torch.Tensor], Field(default=None, exclude=True)]
]

# Pydantic type-hint to exclude numpy array from JSON schema/model dump methods
ExcludedNumpyArray = SkipJsonSchema[
    Annotated[Optional[np.ndarray], Field(default=None, exclude=True)]
]

# Pydantic type-hint to exclude pandas DataFrame from JSON schema/model dump methods
ExcludedDataFrame = SkipJsonSchema[
    Annotated[Optional[pd.DataFrame], Field(default=None, exclude=True)]
]
