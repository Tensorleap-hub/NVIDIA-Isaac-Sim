"""Runtime helpers for the simulation calibration workflow.

This module contains the side-effecting pieces of the pipeline: selecting real
images, embedding images with DINOv2, launching Isaac, and persisting workflow
state and logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class RunArtifact:
    """Captures one evaluated synthetic run and its reusable artifacts."""

    run_id: str
    run_fingerprint: str
    yaml_path: Path
    output_dir: Path
    log_path: Path
    embedding_path: Path
    image_count: int
    flattened_params: dict[str, Any]
    dist_id: str | None
    objective_value: float | None = None
    base_pool_entry_id: str | None = None
    base_pool_lineage: str | None = None


def select_real_image_paths(dataset_root: str | Path, annotation_file: str | Path) -> list[Path]:
    """Resolve LOCO-style annotation entries into concrete local image paths."""
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


def select_real_image_paths_from_csv(
    dataset_root: str | Path,
    annotation_file: str | Path,
    target_csv: str | Path,
    *,
    data_type: str = "real",
    data_type_column: str = "metadata.data_type",
) -> list[Path]:
    """Resolve a Tensorleap sample-export CSV into concrete real image paths.

    `target_csv` is a platform sample export (e.g. a saved insight): each row's
    `sample_id` is `f"{dataset_state}_{image_id}"`, where `image_id` matches the
    `id` field of an entry in `annotation_file`'s LOCO-style `images` list. Only
    rows whose `data_type_column` equals `data_type` are kept, so a CSV mixing
    real and synthetic samples can be used directly as a target definition.
    """
    dataset_root = Path(dataset_root)
    annotation_file = Path(annotation_file)
    target_csv = Path(target_csv)
    payload = json.loads(annotation_file.read_text())
    relative_path_by_image_id = {
        str(item["id"]): item["path"].replace("/dataset/", "", 1) for item in payload["images"]
    }

    with open(target_csv, newline="") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if row.get(data_type_column) == data_type]
    if not rows:
        raise ValueError(f"No rows with {data_type_column}={data_type!r} found in {target_csv}")

    image_paths = []
    seen_image_ids = set()
    unresolved_sample_ids = []
    for row in rows:
        sample_id = row["sample_id"]
        state_prefix = f"{row.get('dataset_state', '')}_"
        image_id = sample_id[len(state_prefix):] if row.get("dataset_state") and sample_id.startswith(state_prefix) else sample_id
        if image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)
        relative_path = relative_path_by_image_id.get(image_id)
        image_path = dataset_root / relative_path if relative_path is not None else None
        if image_path is not None and image_path.exists():
            image_paths.append(image_path)
        else:
            unresolved_sample_ids.append(sample_id)

    if unresolved_sample_ids:
        preview = ", ".join(unresolved_sample_ids[:10])
        print(
            f"select_real_image_paths_from_csv: {len(unresolved_sample_ids)}/{len(rows)} sample id(s) "
            f"from {target_csv} could not be resolved under {dataset_root} (e.g. {preview})"
        )
    return sorted(image_paths)


class DINOv2Embedder:
    """Thin wrapper around Torch Hub DINOv2 inference with disk caching."""

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
        """Embed a list of images, reusing a cache entry when the manifest matches."""
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
        """Load one image and apply the DINOv2 preprocessing pipeline."""
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


# The RF-DETR backbone is a DINOv2 ViT-S/14 (hidden 384, 12 layers, 6 heads,
# patch 14, no register tokens). Its checkpoint stores the encoder weights under
# ``backbone.0.encoder.encoder.*`` with exact HuggingFace ``Dinov2Model`` key
# naming, so we load them straight into a plain ``Dinov2Model`` — no rfdetr /
# transformers-5.x / torch>=2.4 dependency needed, just the fine-tuned encoder.
_RFDETR_ENCODER_PREFIX = "backbone.0.encoder.encoder."
# ``layer_index`` (0-3) selects a depth among 4 evenly-spaced transformer stages,
# mapping onto Dinov2Model's ``hidden_states`` (index 0 = embeddings, 1..12 = layers).
_RFDETR_SCALE_LAYERS = (3, 6, 9, 12)


class RFDETREmbedder:
    """RF-DETR DINOv2 encoder feature extractor using global-average-pool on a selected depth.

    Loads the fine-tuned RF-DETR backbone (a DINOv2 ViT-S/14) into a HuggingFace
    ``Dinov2Model``. ``layer_index`` (0-3) picks one of 4 evenly-spaced transformer
    stages; default 3 uses the deepest. Patch tokens (cls dropped) are mean-pooled
    to a (B, 384) embedding per image.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,  # kept for interface compatibility; unused by the encoder
        layer_index: int,
        device: str,
        resize_size: int,
        image_size: int,
    ):
        from transformers import Dinov2Config, Dinov2Model  # lazy import – only for backend="rfdetr"

        self.device = torch.device(device)
        self.layer_index = layer_index
        self._hidden_state_index = _RFDETR_SCALE_LAYERS[max(0, min(layer_index, len(_RFDETR_SCALE_LAYERS) - 1))]

        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mlp_ratio=4,
            patch_size=14,
            image_size=518,
            num_register_tokens=0,
        )
        model = Dinov2Model(config)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=str(self.device), weights_only=False)
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            encoder_sd = {
                key[len(_RFDETR_ENCODER_PREFIX) :]: value
                for key, value in state_dict.items()
                if key.startswith(_RFDETR_ENCODER_PREFIX)
            }
            if not encoder_sd:
                raise ValueError(
                    f"No RF-DETR encoder weights found under prefix '{_RFDETR_ENCODER_PREFIX}' in {checkpoint_path}"
                )
            result = model.load_state_dict(encoder_sd, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError(
                    "RF-DETR encoder weights did not map cleanly onto Dinov2Model "
                    f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
                )

        self._model = model
        self._model.eval()
        self._model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
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
        """Embed a list of images using the RF-DETR backbone, reusing a cache entry when the manifest matches."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_path.with_suffix(".manifest.json")
        if cache_path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text())
            if cached_manifest == manifest:
                return np.load(cache_path)

        batches = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = torch.stack([self._load_image(p) for p in batch_paths], dim=0).to(self.device)
                outputs = self._model(images, output_hidden_states=True)
                hidden = outputs.hidden_states[self._hidden_state_index]  # (B, 1+N, C)
                patch_tokens = hidden[:, 1:, :]  # drop cls token (no register tokens)
                batches.append(patch_tokens.mean(dim=1).cpu().numpy())  # (B, C)
        embeddings = np.concatenate(batches, axis=0)
        np.save(cache_path, embeddings)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return embeddings

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


_RFDETR_PROJECTOR_PREFIX = "backbone.0.projector."
_RFDETR_SRC_ROOT = str(Path(__file__).resolve().parent.parent / "models" / "rf-detr" / "src")


class RFDETRNeckEmbedder:
    """RF-DETR DINOv2 encoder + trained projector/neck feature extractor.

    Reconstructs the checkpoint's actual multi-scale projector (the module RF-DETR
    calls its P4 stage — the closest functional analogue to YOLO's P3 available in
    this checkpoint, since it only trained a single projector scale) instead of a
    raw, unfused DINOv2 hidden state like ``RFDETREmbedder``. All four
    ``_RFDETR_SCALE_LAYERS`` stages feed the projector, matching how the checkpoint
    was trained; its single output stage is mean-pooled to a (B, out_channels)
    embedding per image.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        resize_size: int,
        image_size: int,
        out_channels: int = 256,
    ):
        from transformers import Dinov2Config, Dinov2Model  # lazy import – only for backend="rfdetr_neck"

        if _RFDETR_SRC_ROOT not in sys.path:
            sys.path.insert(0, _RFDETR_SRC_ROOT)
        from rfdetr.models.backbone.projector import MultiScaleProjector  # noqa: E402

        if image_size % 14 != 0:
            raise ValueError(f"image_size must be a multiple of the ViT-S/14 patch size (14), got {image_size}")
        if not checkpoint_path:
            raise ValueError(
                "RFDETRNeckEmbedder requires a fine-tuned checkpoint_path "
                "(the trained projector has no pretrained-only weights)."
            )

        self.device = torch.device(device)
        self.grid_size = image_size // 14

        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mlp_ratio=4,
            patch_size=14,
            image_size=518,
            num_register_tokens=0,
        )
        encoder = Dinov2Model(config)
        projector = MultiScaleProjector(
            in_channels=[384, 384, 384, 384],
            out_channels=out_channels,
            scale_factors=[1.0],
            num_blocks=3,
            layer_norm=True,
        )

        ckpt = torch.load(checkpoint_path, map_location=str(self.device), weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        encoder_sd = {
            key[len(_RFDETR_ENCODER_PREFIX):]: value
            for key, value in state_dict.items()
            if key.startswith(_RFDETR_ENCODER_PREFIX)
        }
        if not encoder_sd:
            raise ValueError(
                f"No RF-DETR encoder weights found under prefix '{_RFDETR_ENCODER_PREFIX}' in {checkpoint_path}"
            )
        result = encoder.load_state_dict(encoder_sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "RF-DETR encoder weights did not map cleanly onto Dinov2Model "
                f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
            )

        projector_sd = {
            key[len(_RFDETR_PROJECTOR_PREFIX):]: value
            for key, value in state_dict.items()
            if key.startswith(_RFDETR_PROJECTOR_PREFIX)
        }
        if not projector_sd:
            raise ValueError(
                f"No RF-DETR projector weights found under prefix '{_RFDETR_PROJECTOR_PREFIX}' in {checkpoint_path}"
            )
        result = projector.load_state_dict(projector_sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "RF-DETR projector weights did not map cleanly onto MultiScaleProjector "
                f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
            )

        self._encoder = encoder
        self._encoder.eval()
        self._encoder.to(self.device)
        self._projector = projector
        self._projector.eval()
        self._projector.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
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
        """Embed images through the RF-DETR encoder + trained neck, reusing a cache entry when the manifest matches."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_path.with_suffix(".manifest.json")
        if cache_path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text())
            if cached_manifest == manifest:
                return np.load(cache_path)

        batches = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = torch.stack([self._load_image(p) for p in batch_paths], dim=0).to(self.device)
                outputs = self._encoder(images, output_hidden_states=True)
                stage_maps = []
                for hidden_state_index in _RFDETR_SCALE_LAYERS:
                    hidden = outputs.hidden_states[hidden_state_index]  # (B, 1+N, C)
                    patch_tokens = hidden[:, 1:, :]  # drop cls token (no register tokens)
                    b, n, c = patch_tokens.shape
                    stage_maps.append(patch_tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size))
                neck_feat = self._projector(stage_maps)[0]  # (B, out_channels, H, W) — only/finest stage
                batches.append(neck_feat.mean(dim=(2, 3)).cpu().numpy())  # (B, out_channels)
        embeddings = np.concatenate(batches, axis=0)
        np.save(cache_path, embeddings)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return embeddings

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def discover_generated_images(output_dir: Path) -> list[Path]:
    """Recursively collect RGB image files under an output directory."""
    images = []
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return sorted(images)


class ProcessLogStreamer:
    """Mirror Isaac stdout into a per-run log file and UI callback."""

    def __init__(self, process: subprocess.Popen[str], log_path: Path, log_callback):
        self.process = process
        self.log_path = log_path
        self.log_callback = log_callback

    def stream(self) -> None:
        """Drain the child process output until EOF."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as log_file:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                log_file.write(line)
                log_file.flush()
                self.log_callback(line.rstrip())


class StateStore:
    """JSON-backed checkpoint store for workflow resume."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load state from disk, or initialize an empty iteration ledger."""
        if not self.state_path.exists():
            return {"iterations": []}
        return json.loads(self.state_path.read_text())

    def save(self, state: dict[str, Any]) -> None:
        """Persist the current workflow state to disk."""
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def make_cache_key(parts: list[str]) -> str:
    """Create a stable cache key from a sequence of string parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def prepare_output_dir(path: Path, *, clean: bool) -> None:
    """Ensure an output directory exists, optionally replacing it first."""
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
    seed: int | None = None,
    seeds: list[int] | None = None,
    capture_mode: str | None = None,
) -> None:
    """Launch one Isaac SDG job and stream its logs into the workflow."""
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
    ]
    if seeds is not None:
        # Episode mode: one Isaac session renders every seed; the SDG writes
        # <output_dir>/<yaml-stem>_seed<S>/ per episode and retries bad layouts
        # in-process (seed + k*1000).
        command.extend(["--seeds", " ".join(str(s) for s in seeds), "--out_root", str(output_dir)])
    else:
        command.extend(["--data_dir", str(output_dir)])
    if capture_mode is not None:
        command.extend(["--capture_mode", str(capture_mode)])
    if num_frames_override is not None:
        command.extend(["--num_frames", str(num_frames_override)])
    if seed is not None:
        command.extend(["--seed", str(seed)])

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
