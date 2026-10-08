"""Configuración final de Eikonorasís CIFAR-10."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES_ES = (
    "Avión", "Automóvil", "Ave", "Gato", "Ciervo",
    "Perro", "Rana", "Caballo", "Barco", "Camión",
)

@dataclass(frozen=True)
class AppConfig:
    model_path: Path = PROJECT_ROOT / "model" / "model_final.keras"
    logo_path: Path = PROJECT_ROOT / "assets" / "logo.png"
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_upload_bytes: int = 5 * 1024 * 1024
    top_k: int = 3
    target_size: tuple[int, int] = (160, 160)
