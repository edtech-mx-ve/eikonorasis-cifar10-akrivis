"""Selección acotada y reproducible para un piloto de entrenamiento en CPU."""
from __future__ import annotations

import numpy as np

from .transfer_data import balanced_sample_indices, validate_rgb_batch


def validate_pilot_limits(
    train_per_class: int, validation_per_class: int, epochs: int
) -> None:
    """Limita el piloto; el entrenamiento completo usa otro ejecutable."""
    for name, value, lower, upper in (
        ("train_per_class", train_per_class, 2, 500),
        ("validation_per_class", validation_per_class, 1, 100),
        ("epochs", epochs, 1, 5),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} debe ser entero.")
        if not lower <= value <= upper:
            raise ValueError(f"{name} debe estar entre {lower} y {upper} en el piloto.")


def select_balanced_subset(
    images: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve imágenes, etiquetas y sus índices originales, sin modificar entradas."""
    validate_rgb_batch(images, labels)
    y = np.asarray(labels).reshape(-1)
    original = np.asarray(source_indices)
    if original.ndim != 1 or not np.issubdtype(original.dtype, np.integer):
        raise TypeError("Los índices originales deben ser un vector de enteros.")
    if original.size != len(y):
        raise ValueError("La cantidad de índices no coincide con las etiquetas.")
    if np.unique(original).size != original.size:
        raise ValueError("Hay índices originales duplicados.")
    if (original < 0).any() or (original >= 50_000).any():
        raise ValueError("Los índices deben pertenecer al desarrollo oficial CIFAR-10.")
    chosen = balanced_sample_indices(y, per_class=per_class, seed=seed)
    return images[chosen], y[chosen], original[chosen].astype(np.int64, copy=False)
