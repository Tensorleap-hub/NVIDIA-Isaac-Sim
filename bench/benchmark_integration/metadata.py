from __future__ import annotations

import math
from pathlib import Path
import sys

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

_BENCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BENCH_DIR))
from convergence.config import THETA_KEYS


@tensorleap_metadata("data_type", DatasetMetadataType.string)
def data_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx]["data_type"]


@tensorleap_metadata("simulation_type", DatasetMetadataType.string)
def simulation_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx].get("simulation_type", "")


@tensorleap_metadata("theta")
def theta_metadata(idx: str, preprocess: PreprocessResponse) -> dict:
    record = preprocess.data[idx]
    if record["data_type"] == "real":
        return {k: float("nan") for k in THETA_KEYS}
    return {k: float(record[k]) for k in THETA_KEYS}
