"""Persistencia y reportes aislados por ejecución del Sprint 2."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from .config import CLASS_NAMES_ES
from .transfer_model import validate_probabilities


def new_run_directory(parent: Path, smoke_test: bool) -> Path:
    """Crea una carpeta única: nunca borra ni reutiliza experimentos previos."""
    parent.mkdir(parents=True, exist_ok=True)
    prefix = "smoke" if smoke_test else "train"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    result = parent / f"{prefix}_{stamp}_{uuid4().hex[:8]}"
    result.mkdir(exist_ok=False)
    return result


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    """Escribe JSON nuevo; rechaza sobrescritura y números no finitos."""
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def file_digest(path: Path) -> str:
    """Calcula SHA-256 por bloques para un archivo local."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, object]:
    """Registra versiones relevantes sin listar rutas, cuentas ni secretos."""
    packages: dict[str, str] = {}
    for name in ("tensorflow", "keras", "numpy", "h5py", "matplotlib", "keras-tuner"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {"python": platform.python_version(), "os": platform.system(), "packages": packages}


def save_curves(history: Mapping[str, Sequence[float]], directory: Path) -> None:
    """Guarda dos gráficas separadas: pérdida y accuracy, sin estilos impuestos."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for metric, title, label in (
        ("loss", "Pérdida durante el entrenamiento", "Pérdida"),
        ("accuracy", "Accuracy durante el entrenamiento", "Accuracy"),
    ):
        path = directory / f"{metric}.png"
        if path.exists():
            raise FileExistsError(path)
        fig = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        epochs = list(range(1, len(history[metric]) + 1))
        ax.plot(epochs, history[metric], marker="o", label="Entrenamiento")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label="Validación")
        ax.set(title=title, xlabel="Época", ylabel=label)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=140)
        fig.clear()


def write_sample_predictions(
    path: Path, labels: np.ndarray, probabilities: np.ndarray
) -> None:
    """Exporta inferencias de validación; no representa la prueba de 30 imágenes."""
    y = np.asarray(labels)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Se requieren etiquetas enteras unidimensionales.")
    if y.size == 0 or (y < 0).any() or (y >= 10).any():
        raise ValueError("Etiquetas de muestra fuera de 0..9.")
    validate_probabilities(probabilities, y.size)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "expected", "predicted", "score_softmax", "correct"])
        for i, (expected, scores) in enumerate(zip(y, probabilities, strict=True)):
            predicted = int(np.argmax(scores))
            writer.writerow([
                i, CLASS_NAMES_ES[int(expected)], CLASS_NAMES_ES[predicted],
                float(scores[predicted]), predicted == int(expected)
            ])
