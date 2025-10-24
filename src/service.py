# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import re
from rich.console import Console
from tqdm import tqdm

from .infra import init_model, save_text, out_path_for
from .vosk_service import VoskService

console = Console()

_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.U)
_CYR_RE        = re.compile(r"[А-Яа-яЁё]", re.U)

def looks_bad(s: str) -> bool:
    if not s or not s.strip():
        return True
    txt = s.strip()
    if _PUNCT_ONLY_RE.match(txt):
        return True
    letters = sum(ch.isalpha() for ch in txt)
    if letters < max(5, int(len(txt)*0.15)):
        return True
    if not _CYR_RE.search(txt):
        return True
    return False

def _presets(word_ts: bool) -> List[Dict[str, Any]]:
    return [
        dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            beam_size=5, temperature=0.0,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-0.3,
            no_speech_threshold=0.8,
            condition_on_previous_text=False,
            initial_prompt=(
                "Телефонный разговор на русском языке. Нормальная речь, без музыки и шумов. "
                "Не использовать лишние восклицательные знаки."
            ),
            suppress_blank=True,
            word_timestamps=word_ts,
        ),
        dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=800),
            beam_size=1, temperature=0.0,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.5,
            no_speech_threshold=0.85,
            condition_on_previous_text=False,
            initial_prompt="Телефонный разговор на русском языке.",
            suppress_blank=True,
            word_timestamps=word_ts,
        ),
        dict(
            vad_filter=False,
            beam_size=5, temperature=0.0,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.5,
            no_speech_threshold=0.9,
            condition_on_previous_text=False,
            initial_prompt="Разговор на русском языке.",
            suppress_blank=True,
            word_timestamps=word_ts,
        ),
    ]

class TranscriberService:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.engine = cfg.get("engine", "auto")
        self.model = init_model(cfg)

    def _whisper_try(self, audio_path: Path, params: Dict[str, Any]) -> Tuple[str, float]:
        language = self.cfg.get("language", "ru")
        segments, info = self.model.transcribe(str(audio_path), language=language, **params)
        lines = []
        for seg in tqdm(segments, desc=f"Обработка: {audio_path.name}", unit="seg"):
            if seg.text:
                lines.append(seg.text.strip())
        text = "\n".join(line for line in lines if line)
        dur = getattr(info, "duration", 0.0) if info is not None else 0.0
        return text, dur

    def transcribe_file(self, audio_path: Path) -> Tuple[str, float]:
        word_ts = bool(self.cfg.get("word_timestamps", False))

        if self.engine == "vosk":
            v = VoskService()
            return v.transcribe_wav16k(audio_path)

        if self.engine == "whisper":
            for i, params in enumerate(_presets(word_ts), start=1):
                console.log(f"[cyan]Whisper пресет {i}[/cyan]")
                text, dur = self._whisper_try(audio_path, params)
                if not looks_bad(text):
                    console.log(f"[green]Принят пресет {i}[/green]")
                    return text, dur
            return self._whisper_try(audio_path, dict(
                vad_filter=False, beam_size=1, temperature=0.0,
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.9, condition_on_previous_text=False,
                suppress_blank=True, word_timestamps=word_ts
            ))

        # AUTO
        for i, params in enumerate(_presets(word_ts), start=1):
            console.log(f"[cyan]Пробую пресет {i}[/cyan]: {params.get('beam_size')} beam, VAD={'on' if params.get('vad_filter') else 'off'}")
            text, dur = self._whisper_try(audio_path, params)
            if not looks_bad(text):
                console.log(f"[green]Принят пресет {i}[/green] — текст выглядит осмысленным.")
                return text, dur
            else:
                console.log(f"[yellow]Отклонён пресет {i}[/yellow] — текст некачественный, пробую дальше.")

        console.log("[red]Все пресеты Whisper дали слабый результат.[/red]")
        try:
            text, dur = self._whisper_try(audio_path, dict(
                vad_filter=False, beam_size=1, temperature=0.0,
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.9, condition_on_previous_text=False,
                suppress_blank=True, word_timestamps=word_ts
            ))
        except Exception:
            text, dur = "", 0.0

        if looks_bad(text):
            try:
                console.log("[magenta]Пробую Vosk fallback (RU 16kHz mono)[/magenta]")
                v = VoskService()
                vtext, vdur = v.transcribe_wav16k(audio_path)
                if not looks_bad(vtext):
                    console.log("[green]Vosk дал хороший результат — используем его.[/green]")
                    return vtext, vdur if vdur else dur
                else:
                    console.log("[yellow]Vosk тоже дал слабый текст — сохраняю Whisper как есть.[/yellow]")
            except Exception as e:
                console.log(f"[red]Vosk fallback не сработал[/red]: {e}")
        return text, dur

    def transcribe_and_save(self, audio_path: Path) -> Path:
        text, dur = self.transcribe_file(audio_path)
        out_path = out_path_for(audio_path)
        save_text(out_path, text)
        console.log(f"[green]Готово[/green]: {audio_path.name} → {out_path.name} (длительность ~ {dur:.1f}с)")
        return out_path
