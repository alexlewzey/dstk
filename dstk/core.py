""""""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import *
import numpy as np

DateLike = Union[str, datetime, datetime.date]
PathOrStr = Union[str, Path]
OptPathOrStr = Optional[Union[str, Path]]
DfOrSer = Union[pd.DataFrame, pd.Series]
DfOrArr = Union[pd.DataFrame, np.ndarray]
OptSeq = Optional[Sequence]
