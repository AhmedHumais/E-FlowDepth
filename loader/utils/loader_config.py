
from .dsec_utils import RepresentationType
from enum import Enum
from typing import Union
from dataclasses import dataclass

# ======================================================================================
# Constants
# ======================================================================================

DEFAULT_DELTA_T_MS = 100
DEFAULT_NUM_BINS = 15

# ======================================================================================
# Task definition
# ======================================================================================

class DSECTask(str, Enum):
    FLOW = "flow"
    DISPARITY = "disparity"
    BOTH = "both"


@dataclass(frozen=True)
class LoaderConfig:
    representation_type: RepresentationType = RepresentationType.VOXEL
    task: Union[str, DSECTask] = DSECTask.FLOW
    split: str = "test"
    delta_t_ms: int = DEFAULT_DELTA_T_MS
    num_bins: int = DEFAULT_NUM_BINS

    def normalized_task(self) -> DSECTask:
        if isinstance(self.task, DSECTask):
            return self.task
        return DSECTask(self.task.lower())