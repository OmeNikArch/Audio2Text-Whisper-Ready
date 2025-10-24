# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "audio"
OUT_DIR = PROJECT_ROOT / "transcripts"
CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"

def ensure_dirs() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def init_model(cfg: Dict[str, Any]):
    from faster_whisper import WhisperModel
    model_size = cfg.get("model_size", "small")
    compute_type = cfg.get("compute_type", "int8")
    device = "cuda" if _has_cuda() else "cpu"
    if device == "cpu" and compute_type != "int8":
        console.log("[yellow]CPU-режим обнаружен → compute_type принудительно 'int8'[/yellow]")
        compute_type = "int8"
    console.log(f"[bold]Загрузка модели[/bold]: size={model_size}, device={device}, compute={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def list_audio_files(only_ext: tuple = (".mp3", ".wav", ".m4a", ".mp4", ".ogg")) -> list[Path]:
    files = []
    for ext in only_ext:
        files.extend(AUDIO_DIR.glob(f"*{ext}"))
    return sorted(files)

def out_path_for(audio_path: Path) -> Path:
    base = audio_path.stem
    return OUT_DIR / f"{base}.txt"

def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write((text or "").strip() + "\n")
