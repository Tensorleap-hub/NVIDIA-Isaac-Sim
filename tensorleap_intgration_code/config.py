import os
import yaml
from typing import Any, Dict


def abs_path_from_root(path: str) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, path)


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "project_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# Flat lookups derived from config
COCO_ID_TO_IDX: Dict[int, int] = {c["coco_id"]: c["idx"] for c in CONFIG["categories"]}
IDX_TO_NAME: Dict[int, str] = {c["idx"]: c["name"] for c in CONFIG["categories"]}
CLASS_NAMES = [IDX_TO_NAME[i] for i in range(len(CONFIG["categories"]))]
