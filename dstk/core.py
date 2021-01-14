""""""
from datetime import datetime
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd

PathOrStr = Union[str, Path]
OptPathOrStr = Optional[Union[str, Path]]
OptSeq = Optional[Sequence]

DateLike = Union[str, datetime, datetime.date]

DfOrSer = Union[pd.DataFrame, pd.Series]
DfOrArr = Union[pd.DataFrame, np.ndarray]
