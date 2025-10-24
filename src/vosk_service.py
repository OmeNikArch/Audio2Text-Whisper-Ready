# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import json, wave, os
from vosk import Model, KaldiRecognizer
from rich.console import Console

console = Console()

def _default_model_path() -> Path:
    env = os.environ.get("VOSK_MODEL_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / "models" / "vosk-model-small-ru-0.22"

class VoskService:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else _default_model_path()
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена: {self.model_dir}\n"
                "Скачайте модель и задайте VOSK_MODEL_PATH или поместите в ~/models/."
            )
        console.log(f"[bold]Загрузка Vosk модели[/bold]: {self.model_dir}")
        self.model = Model(str(self.model_dir))

    def transcribe_wav16k(self, wav_path: Path) -> Tuple[str, float]:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise AssertionError("Ожидается моно (1 канал). Сделайте препроцессинг.")
            if wf.getframerate() != 16000:
                raise AssertionError("Ожидается 16 кГц. Сделайте препроцессинг.")
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            parts = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if "text" in res and res["text"].strip():
                        parts.append(res["text"].strip())
            res = json.loads(rec.FinalResult())
            if "text" in res and res["text"].strip():
                parts.append(res["text"].strip())
            text = "\n".join(p for p in parts if p)
            duration = wf.getnframes() / float(wf.getframerate()) if wf.getframerate() else 0.0
            return text, duration
