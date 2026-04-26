from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess

_BENCH_DIR = Path(__file__).parent.parent
_CONVERGENCE_DIR = _BENCH_DIR / "convergence"

# Import config values from the convergence package
import sys
sys.path.insert(0, str(_BENCH_DIR))
from convergence.config import DATA_ROOT, THETA_KEYS, N_REAL_IMAGES

_REAL_DIR = DATA_ROOT / "real"
_TL_SEED_DIR = DATA_ROOT / "tl_seed"
_METADATA_PATH = _TL_SEED_DIR / "metadata.csv"


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    real_records = []
    for img_path in sorted(_REAL_DIR.glob("*.png")):
        real_records.append({"image_path": str(img_path), "data_type": "real", "simulation_type": ""})

    meta_df = pd.read_csv(_METADATA_PATH)
    synth_records = []
    for _, row in meta_df.iterrows():
        record = {
            "image_path": str(row["image_path"]),
            "data_type": "synth",
            "simulation_type": "simulation_1",
        }
        for k in THETA_KEYS:
            record[k] = float(row[k])
        synth_records.append(record)

    split = int(len(real_records) * 0.8)
    train_records, val_records = real_records[:split], real_records[split:]
    train_ids = [f"real_{i:04d}" for i in range(len(train_records))]
    val_ids = [f"real_{i:04d}" for i in range(len(train_records), len(real_records))]
    synth_ids = [f"synth_{i:06d}" for i in range(len(synth_records))]

    return [
        PreprocessResponse(
            data={sid: r for sid, r in zip(train_ids, train_records)},
            sample_ids=train_ids,
            state=DataStateType.training,
        ),
        PreprocessResponse(
            data={sid: r for sid, r in zip(val_ids, val_records)},
            sample_ids=val_ids,
            state=DataStateType.validation,
        ),
        PreprocessResponse(
            data={sid: r for sid, r in zip(synth_ids, synth_records)},
            sample_ids=synth_ids,
            state=DataStateType.additional,
        ),
    ]
