from typing import Dict, List, Protocol

import numpy as np

from .contracts import SingleSimulationData


class SampleDispatcher(Protocol):
    def generate(
        self,
        sim_data_by_dist: Dict[str, List[SingleSimulationData]],
    ) -> Dict[str, List[str]]:
        ...

    def collect_ls(
        self,
        sample_ids_by_dist: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        ...
