"""Carga, validación y partición reproducible de CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import CLASS_NAMES_ES, DataConfig

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class DatasetBundle:
    """Contenedor inmutable de las particiones de CIFAR-10."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray

def _as_1d_labels(labels: np.ndarray) -> np.ndarray:
    """Convierte etiquetas a un vector unidimensional de enteros."""
    arr = np.asarray(labels)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Las etiquetas deben ser 1D o tener forma (n, 1).")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Las etiquetas deben ser enteras.")
    return arr.astype(np.int64, copy=False)

def validate_labels(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Valida rango, clases presentes y tipo de etiquetas."""
    y = _as_1d_labels(labels)
    if y.size == 0:
        raise ValueError("El vector de etiquetas no puede estar vacío.")
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError(
            f"Las etiquetas deben estar en el rango [0, {num_classes - 1}]."
        )
    missing = sorted(set(range(num_classes)) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(f"Faltan clases en las etiquetas: {missing}")
    return y

def stratified_train_val_indices(
    labels: np.ndarray,
    validation_per_class: int,
    seed: int,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera índices train/validation estratificados y reproducibles."""
    if validation_per_class <= 0:
        raise ValueError("validation_per_class debe ser mayor que cero.")

    y = validate_labels(labels, num_classes=num_classes)
    rng = np.random.default_rng(seed)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_id in range(num_classes):
        class_idx = np.flatnonzero(y == class_id)
        if class_idx.size <= validation_per_class:
            raise ValueError(
                f"La clase {class_id} no tiene suficientes ejemplos: "
                f"{class_idx.size} <= {validation_per_class}."
            )
        shuffled = rng.permutation(class_idx)
        val_parts.append(shuffled[:validation_per_class])
        train_parts.append(shuffled[validation_per_class:])

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    validate_split_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        total_size=y.size,
    )
    return train_idx, val_idx

def validate_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    total_size: int,
) -> None:
    """Verifica cobertura, unicidad, rango y ausencia de solapamiento."""
    if total_size <= 0:
        raise ValueError("total_size debe ser mayor que cero.")

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    for name, idx in (("train", train_idx), ("validation", val_idx)):
        if idx.ndim != 1:
            raise ValueError(f"Los índices de {name} deben ser unidimensionales.")
        if idx.size != np.unique(idx).size:
            raise ValueError(f"Hay índices duplicados en {name}.")
        if idx.size and (idx.min() < 0 or idx.max() >= total_size):
            raise ValueError(f"Hay índices fuera de rango en {name}.")

    overlap = np.intersect1d(train_idx, val_idx)
    if overlap.size:
        raise ValueError(
            f"Train y validation se solapan en {overlap.size} índices."
        )

    if train_idx.size + val_idx.size != total_size:
        raise ValueError(
            "Train y validation no cubren exactamente el conjunto de desarrollo."
        )

def save_split_indices(
    path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> None:
    """Guarda la partición para reproducibilidad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        seed=np.asarray([seed], dtype=np.int64),
    )
    LOGGER.info("Partición guardada en %s", path)

def load_split_indices(path: Path, total_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Carga y valida una partición previamente guardada."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de partición: {path}")

    with np.load(path, allow_pickle=False) as data:
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]

    validate_split_indices(train_idx, val_idx, total_size)
    return train_idx, val_idx

def load_cifar10(config: DataConfig | None = None) -> DatasetBundle:
    """Descarga/carga CIFAR-10 y construye train/validation/test.

    La descarga se realiza mediante keras.datasets.cifar10 la primera vez.
    El test oficial de CIFAR-10 se mantiene intacto para evaluación final.
    """
    cfg = config or DataConfig()

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Ejecuta: pip install -r requirements.txt"
        ) from exc

    LOGGER.info("Cargando CIFAR-10...")
    (x_dev, y_dev), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_dev = validate_labels(y_dev, cfg.num_classes)
    y_test = validate_labels(y_test, cfg.num_classes)

    if x_dev.shape != (50_000, 32, 32, 3):
        raise ValueError(f"Forma inesperada para x_dev: {x_dev.shape}")
    if x_test.shape != (cfg.expected_test_size, 32, 32, 3):
        raise ValueError(f"Forma inesperada para x_test: {x_test.shape}")

    if cfg.split_path.exists():
        train_idx, val_idx = load_split_indices(cfg.split_path, y_dev.size)
        LOGGER.info("Partición reproducible cargada desde disco.")
    else:
        train_idx, val_idx = stratified_train_val_indices(
            labels=y_dev,
            validation_per_class=cfg.validation_per_class,
            seed=cfg.seed,
            num_classes=cfg.num_classes,
        )
        save_split_indices(cfg.split_path, train_idx, val_idx, cfg.seed)

    bundle = DatasetBundle(
        x_train=x_dev[train_idx],
        y_train=y_dev[train_idx],
        x_val=x_dev[val_idx],
        y_val=y_dev[val_idx],
        x_test=x_test,
        y_test=y_test,
        train_indices=train_idx,
        val_indices=val_idx,
    )
    validate_bundle(bundle, cfg)
    return bundle

def validate_bundle(bundle: DatasetBundle, config: DataConfig | None = None) -> None:
    """Valida tamaños, clases y consistencia de las particiones."""
    cfg = config or DataConfig()

    expected = {
        "train": cfg.expected_train_size,
        "validation": cfg.expected_validation_size,
        "test": cfg.expected_test_size,
    }
    actual = {
        "train": bundle.y_train.size,
        "validation": bundle.y_val.size,
        "test": bundle.y_test.size,
    }
    if actual != expected:
        raise ValueError(f"Tamaños inesperados. Esperado={expected}, actual={actual}")

    for split_name, labels in (
        ("train", bundle.y_train),
        ("validation", bundle.y_val),
        ("test", bundle.y_test),
    ):
        y = validate_labels(labels, cfg.num_classes)
        counts = np.bincount(y, minlength=cfg.num_classes)
        LOGGER.info("%s: %s", split_name, counts.tolist())

def dataset_summary(bundle: DatasetBundle) -> dict[str, object]:
    """Devuelve un resumen serializable del dataset preparado."""
    return {
        "train": int(bundle.y_train.size),
        "validation": int(bundle.y_val.size),
        "test": int(bundle.y_test.size),
        "classes": list(CLASS_NAMES_ES),
        "train_per_class": np.bincount(bundle.y_train, minlength=10).tolist(),
        "validation_per_class": np.bincount(bundle.y_val, minlength=10).tolist(),
        "test_per_class": np.bincount(bundle.y_test, minlength=10).tolist(),
    }
