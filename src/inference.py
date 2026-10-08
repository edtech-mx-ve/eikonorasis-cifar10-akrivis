"""Inferencia final de Eikonorasís CIFAR-10."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from .config import CLASS_NAMES_ES
from .preprocessing import prepare_image

@dataclass(frozen=True)
class Prediction:
    class_index: int
    label: str
    probability: float

class Cifar10Classifier:
    def __init__(
        self,
        model_path: Path,
        class_names: tuple[str, ...] = CLASS_NAMES_ES,
    ) -> None:
        self._model_path = Path(model_path)
        self._class_names = class_names
        self._model: Any | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def load(self) -> None:
        if not self._model_path.is_file():
            raise FileNotFoundError(f"No existe el modelo: {self._model_path}")
        from tensorflow import keras
        self._model = keras.models.load_model(self._model_path)

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("El modelo no está cargado.")
        if not 1 <= top_k <= len(self._class_names):
            raise ValueError("top_k fuera de rango.")
        batch = prepare_image(image, (160, 160))
        raw = np.asarray(self._model.predict(batch, verbose=0), dtype=np.float64)
        if raw.shape != (1, len(self._class_names)):
            raise RuntimeError(f"Salida inesperada: {raw.shape}")
        probabilities = raw[0]
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Probabilidades no finitas.")
        total = float(probabilities.sum())
        if total <= 0:
            raise RuntimeError("Probabilidades inválidas.")
        probabilities /= total
        indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            Prediction(int(i), self._class_names[int(i)], float(probabilities[int(i)]))
            for i in indices
        ]
