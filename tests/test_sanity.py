# -*- coding: utf-8 -*-
import shutil
from pathlib import Path
import yaml

def test_ffmpeg_installed():
    assert shutil.which("ffmpeg") is not None, "ffmpeg должен быть установлен (sudo apt install ffmpeg)."

def test_config_load():
    cfg_path = Path(__file__).resolve().parents[1] / "src" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "model_size" in cfg and "language" in cfg

def test_dirs():
    root = Path(__file__).resolve().parents[1]
    assert (root / "transcripts").exists()
    assert (root / "audio").exists()
