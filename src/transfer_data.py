"""Validación de entradas y reutilización segura de la partición del Sprint 1."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import DataConfig
from .data import validate_labels, validate_split_indices


def validate_rgb_batch(images: np.ndarray, labels: np.ndarray) -> None:
    """Valida imágenes CIFAR RGB uint8 sin normalizar y etiquetas 0..9.

    No transforma entradas: la normalización pertenece al modelo serializado.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("Las imágenes deben ser un arreglo NumPy.")
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("Se esperan imágenes con forma (N, 32, 32, 3).")
    if images.dtype != np.uint8:
        raise TypeError("Se requiere uint8 sin normalizar; no dividir entre 255.")
    y = validate_labels(labels, 10)
    if images.shape[0] != y.size:
        raise ValueError("El número de imágenes y etiquetas no coincide.")


def balanced_sample_indices(
    labels: np.ndarray, per_class: int, seed: int
) -> np.ndarray:
    """Selecciona ejemplos por clase sin reemplazo; no altera las entradas."""
    if type(per_class) is not int or per_class < 1:
        raise ValueError("per_class debe ser un entero positivo.")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed debe ser un entero entre 0 y 2**31-1.")
    y = validate_labels(labels, 10)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(10):
        indices = np.flatnonzero(y == class_id)
        if indices.size < per_class:
            raise ValueError(f"La clase {class_id} tiene pocos ejemplos.")
        selected.append(rng.choice(indices, per_class, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def audit_saved_split(path: Path, cfg: DataConfig) -> None:
    """Comprueba el archivo existente antes de cargarlo con el módulo anterior.

    Conserva los índices ya aprobados: no regenera ni sobrescribe la partición.
    """
    if not path.is_file():
        raise FileNotFoundError(
            "Falta artifacts/split_indices.npz. Ejecuta primero "
            "python scripts/prepare_data.py."
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"train_idx", "val_idx", "seed"}
            if not required.issubset(archive.files):
                raise ValueError("La partición no contiene las claves necesarias.")
            train, val, seed = (
                archive["train_idx"], archive["val_idx"], archive["seed"]
            )
    except (OSError, EOFError) as exc:
        raise ValueError("No se pudo leer la partición guardada.") from exc
    for indices in (train, val):
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Los índices guardados deben ser enteros.")
    validate_split_indices(train, val, 50_000)
    if train.size != cfg.expected_train_size or val.size != cfg.expected_validation_size:
        raise ValueError("La partición guardada no corresponde a 45,000/5,000.")
    if seed.size != 1 or not np.issubdtype(seed.dtype, np.integer):
        raise ValueError("Semilla de partición inválida.")
    if int(seed.item()) != cfg.seed:
        raise ValueError("La semilla guardada difiere de la configuración del Sprint 1.")
