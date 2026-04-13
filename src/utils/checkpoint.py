# src/utils/checkpoint.py
import os
import torch

BASE_MODEL_DIR = "models"
BASE_LOG_DIR = "logs"

_current_run_id: int | None = None


def _get_latest_run_id(round: str, iter_id: str) -> int | None:
    dir_path = os.path.join(BASE_MODEL_DIR, round, iter_id)
    if not os.path.exists(dir_path):
        return None
    run_ids = [int(f) for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f)) and f.isdigit()]
    return max(run_ids) if run_ids else None


def _get_current_run_id(run_config) -> int:
    global _current_run_id
    if _current_run_id is None:
        latest = _get_latest_run_id(run_config.round, run_config.iter_id)
        _current_run_id = 0 if latest is None else latest + 1
    return _current_run_id


def get_save_paths(run_config) -> tuple[str, str]:
    """Returns (model_path, checkpoint_path) under /models/{round}/{iter_id}/{run_id}/"""
    run_id = _get_current_run_id(run_config)
    run_dir = os.path.join(BASE_MODEL_DIR, run_config.round, run_config.iter_id, str(run_id))
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "model.pt"), os.path.join(run_dir, "checkpoint.pt")


def get_latest_paths(run_config) -> tuple[str, str] | None:
    """Returns (model_path, checkpoint_path) for the latest existing run, or None."""
    latest = _get_latest_run_id(run_config.round, run_config.iter_id)
    if latest is None:
        return None
    run_dir = os.path.join(BASE_MODEL_DIR, run_config.round, run_config.iter_id, str(latest))
    return os.path.join(run_dir, "model.pt"), os.path.join(run_dir, "checkpoint.pt")


def get_log_path(run_config) -> str | None:
    if run_config.dry_run:
        return None
    run_id = _get_current_run_id(run_config)
    path = os.path.join(BASE_LOG_DIR, run_config.round, run_config.iter_id, str(run_id))
    os.makedirs(path, exist_ok=True)
    return path


def _safe_save(obj, path: str):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(model, optimizer, run_config) -> int:
    if run_config.dry_run:
        print("Dry run: skipping checkpoint load.")
        return 0
    paths = get_latest_paths(run_config)
    if paths is None:
        print(f"No checkpoint found for {run_config.round}/{run_config.iter_id}. Starting fresh.")
        return 0

    model_path, checkpoint_path = paths
    if not os.path.exists(checkpoint_path):
        print(f"Run dir exists but no checkpoint.pt found at {checkpoint_path}")
        return 0

    model.load_state_dict(torch.load(model_path))
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    print(f"Resumed from step {step} | model: {model_path} | checkpoint: {checkpoint_path}")
    return step


def save_checkpoint(model, optimizer, step, run_config):
    if run_config.dry_run:
        return
    model_path, checkpoint_path = get_save_paths(run_config)
    _safe_save(model.state_dict(), model_path)
    _safe_save({"optimizer": optimizer.state_dict(), "step": step}, checkpoint_path)
    print(f"Saved model to {model_path} | checkpoint to {checkpoint_path}")