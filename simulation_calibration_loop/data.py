from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class RunArtifact:
    run_id: str
    yaml_path: Path
    output_dir: Path
    log_path: Path
    embedding_path: Path
    image_count: int
    flattened_params: dict[str, Any]
    optuna_trial_number: int | None
    objective_value: float | None = None


def select_real_image_paths(dataset_root: str | Path, annotation_file: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    annotation_file = Path(annotation_file)
    payload = json.loads(annotation_file.read_text())
    image_paths = []
    for item in payload["images"]:
        relative_path = item["path"].replace("/dataset/", "", 1)
        image_path = dataset_root / relative_path
        if image_path.exists():
            image_paths.append(image_path)
    return sorted(image_paths)


class DINOv2Embedder:
    def __init__(self, repo: str, model_name: str, device: str, image_size: int, resize_size: int):
        self.repo = repo
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = torch.hub.load(repo, model_name)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def embed_paths(
        self,
        image_paths: list[Path],
        *,
        batch_size: int,
        cache_path: Path,
        manifest: dict[str, Any],
    ) -> np.ndarray:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_path.with_suffix(".manifest.json")
        if cache_path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text())
            if cached_manifest == manifest:
                return np.load(cache_path)

        batches = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                batch = torch.stack([self._load_image(path) for path in batch_paths], dim=0).to(self.device)
                features = self.model(batch)
                batches.append(features.detach().cpu().numpy())
        embeddings = np.concatenate(batches, axis=0)
        np.save(cache_path, embeddings)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return embeddings

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def discover_generated_images(output_dir: Path) -> list[Path]:
    images = []
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return sorted(images)


class ProcessLogStreamer:
    def __init__(self, process: subprocess.Popen[str], log_path: Path, log_callback):
        self.process = process
        self.log_path = log_path
        self.log_callback = log_callback

    def stream(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as log_file:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                log_file.write(line)
                log_file.flush()
                self.log_callback(line.rstrip())


class StateStore:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"iterations": []}
        return json.loads(self.state_path.read_text())

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def make_cache_key(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def prepare_output_dir(path: Path, *, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_isaac_generation(
    *,
    isaac_sim_path: Path,
    script_path: Path,
    yaml_path: Path,
    output_dir: Path,
    log_path: Path,
    headless: bool,
    num_frames_override: int | None,
    log_callback,
) -> None:
    nvjitlink_lib_dir = isaac_sim_path / "exts" / "omni.isaac.ml_archive" / "pip_prebundle" / "nvidia" / "nvjitlink" / "lib"
    env = dict(os.environ)
    if nvjitlink_lib_dir.is_dir():
        current_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{nvjitlink_lib_dir}:{current_ld_library_path}"
            if current_ld_library_path
            else str(nvjitlink_lib_dir)
        )

    command = [
        "./python.sh",
        str(script_path),
        "--config",
        str(yaml_path),
        "--headless",
        "True" if headless else "False",
        "--data_dir",
        str(output_dir),
    ]
    if num_frames_override is not None:
        command.extend(["--num_frames", str(num_frames_override)])

    process = subprocess.Popen(
        command,
        cwd=isaac_sim_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    streamer = ProcessLogStreamer(process=process, log_path=log_path, log_callback=log_callback)
    streamer.stream()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Isaac run failed for {yaml_path.name} with exit code {return_code}. "
            f"See log: {log_path}"
        )
