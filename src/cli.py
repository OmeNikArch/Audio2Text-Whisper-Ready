# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
from rich.console import Console

from .infra import ensure_dirs, load_config, list_audio_files
from .service import TranscriberService

console = Console()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio2text",
        description="Гибридная транскрибация (Whisper GPU + Vosk fallback)."
    )
    p.add_argument("--file", type=str, help="Путь к одному аудиофайлу для транскрибации.")
    p.add_argument("--batch", action="store_true", help="Пакетная обработка всех файлов из папки audio/.")
    p.add_argument("--download-model", action="store_true", help="Только загрузить Whisper модель (прогрев).")
    p.add_argument("--force-small", action="store_true", help="Принудительно использовать модель 'small' (если не хватает VRAM).")
    p.add_argument("--config", type=str, default=None, help="Путь к YAML-конфигу (по умолчанию src/config.yaml).")
    p.add_argument("--engine", choices=["auto", "whisper", "vosk"], default="auto",
                   help="Движок распознавания: auto (по умолчанию), whisper, vosk.")
    return p

def load_config_override(path: str):
    import yaml
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Конфиг не найден: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main_cli() -> int:
    ensure_dirs()
    args = build_argparser().parse_args()
    cfg = load_config() if args.config is None else load_config_override(args.config)

    if args.force_small:
        cfg['model_size'] = 'small'
        console.log("[yellow]--force-small активен[/yellow]: переключаю model_size → 'small'")

    cfg['engine'] = args.engine

    service = TranscriberService(cfg)

    if args.download_model:
        console.log("[bold green]Whisper модель успешно инициализирована[/bold green] — можно запускать транскрибацию.")
        return 0

    if args.file:
        path = Path(args.file)
        if not path.exists():
            console.log(f"[red]Файл не найден[/red]: {path}")
            return 2
        service.transcribe_and_save(path)
        return 0

    if args.batch:
        files = list_audio_files()
        if not files:
            console.log("[yellow]Нет аудиофайлов в папке audio/[/yellow]")
            return 0
        for f in files:
            try:
                service.transcribe_and_save(f)
            except Exception as e:
                console.log(f"[red]Ошибка при обработке {f.name}[/red]: {e}")
        return 0

    console.log("[yellow]Ничего не выбрано. Используйте --file или --batch (или --download-model).[/yellow]")
    return 0
