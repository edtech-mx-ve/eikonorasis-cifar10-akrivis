"""Entrenamiento de la nueva salida y comprobación de persistencia.

El módulo no recibe el conjunto de prueba; únicamente entrenamiento y validación.
"""
from __future__ import annotations

from pathlib import Path
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .transfer_config import TransferConfig
from .transfer_model import (
    parameter_counts, validate_probabilities, weights_digest,
)
from .transfer_reports import save_curves, write_json_new, write_sample_predictions

if TYPE_CHECKING:
    import keras

LOGGER = logging.getLogger(__name__)


def resolve_execution_mode(config: TransferConfig, requested: str | None) -> str:
    """Separa prueba corta, piloto y entrenamiento completo en los reportes."""
    mode = requested if requested is not None else (
        "smoke_test" if config.smoke_test else "frozen_head_training"
    )
    if mode not in ("smoke_test", "pilot_frozen_head_training", "frozen_head_training"):
        raise ValueError("Modo de ejecución desconocido.")
    if config.smoke_test != (mode == "smoke_test"):
        raise ValueError("El modo y smoke_test son incompatibles.")
    return mode


def train_frozen_head(
    model: keras.Model,
    base: keras.Model,
    training_data: Any,
    validation_data: Any,
    check_images: np.ndarray,
    check_labels: np.ndarray,
    config: TransferConfig,
    directory: Path,
    *,
    execution_mode: str | None = None,
    verbose: int = 2,
) -> dict[str, object]:
    """Entrena, verifica base congelada y comprueba recarga de un archivo .keras.

    training_data y validation_data: tf.data.Dataset o tuplas (x, y) para pruebas.
    directory debe ser una carpeta nueva creada exclusivamente para este ensayo.
    Solo el checkpoint de este ensayo puede actualizarse cuando mejora val_loss.
    """
    mode = resolve_execution_mode(config, execution_mode)
    if type(verbose) is not int or verbose not in (0, 1, 2):
        raise ValueError("verbose debe ser 0, 1 o 2.")
    import keras

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    checkpoint = directory / "model.keras"
    history_path = directory / "history.csv"
    if checkpoint.exists() or history_path.exists():
        raise FileExistsError("No se reutiliza una carpeta de entrenamiento anterior.")
    if base.trainable or base.trainable_weights:
        raise ValueError("El Sprint 2 requiere la base completamente congelada.")

    base_before = weights_digest(base.weights)
    head_before = weights_digest(model.get_layer("class_probabilities").weights)
    write_json_new(directory / "model_parameters.json", parameter_counts(model))
    with (directory / "model_summary.txt").open("x", encoding="utf-8") as stream:
        def record_summary(text: str, line_break: bool = True) -> None:
            """Registra el resumen sin mantener un archivo abierto al entrenar."""
            stream.write(text + ("\n" if line_break else ""))
        model.summary(print_fn=record_summary, show_trainable=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min", save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", min_delta=0.0,
            patience=config.patience, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_path), append=False),
        keras.callbacks.TerminateOnNaN(),
    ]
    options = dict(
        validation_data=validation_data,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    started = time.perf_counter()
    if isinstance(training_data, tuple):
        history_obj = model.fit(
            training_data[0], training_data[1], batch_size=config.batch_size,
            shuffle=True, **options
        )
    else:
        # El tf.data.Dataset ya aplica shuffle antes de batch; no duplicarlo.
        history_obj = model.fit(training_data, shuffle=False, **options)
    elapsed = time.perf_counter() - started
    history = {
        key: [float(value) for value in series]
        for key, series in history_obj.history.items()
    }
    for required in ("loss", "accuracy", "val_loss", "val_accuracy"):
        series = history.get(required)
        if not series or not np.isfinite(series).all():
            raise RuntimeError(f"Historia ausente o no finita: {required}.")
    base_unchanged = base_before == weights_digest(base.weights)
    head_changed = head_before != weights_digest(model.get_layer("class_probabilities").weights)
    if not base_unchanged or not head_changed:
        raise RuntimeError("Falló la comprobación de base congelada o salida entrenada.")
    if not checkpoint.is_file():
        raise RuntimeError("No se generó un checkpoint válido.")

    original = np.asarray(keras.ops.convert_to_numpy(model(check_images, training=False)))
    validate_probabilities(original, len(check_images))
    restored = keras.models.load_model(str(checkpoint), compile=False, safe_mode=True)
    reloaded = np.asarray(keras.ops.convert_to_numpy(restored(check_images, training=False)))
    validate_probabilities(reloaded, len(check_images))
    if not np.allclose(original, reloaded, rtol=1e-4, atol=1e-6):
        raise RuntimeError("Las predicciones cambian al guardar y recargar el modelo.")
    # El siguiente sprint compilará con un optimizador nuevo al hacer ajuste fino.
    best_epoch = int(np.argmin(history["val_loss"]))
    write_sample_predictions(directory / "sample_predictions.csv", check_labels, reloaded)
    save_curves(history, directory)
    result: dict[str, object] = {
        "status": "completed",
        "mode": mode,
        "epochs_completed": len(history["loss"]),
        "best_epoch": best_epoch + 1,
        "selection_metric": "val_loss",
        "validation_loss": history["val_loss"][best_epoch],
        "validation_accuracy": history["val_accuracy"][best_epoch],
        "fit_seconds": elapsed,
        "base_unchanged": base_unchanged,
        "head_changed": head_changed,
        "reload_consistent": True,
        "test_evaluated": False,
        "target_92_verified": False,
        "eligible_for_fine_tuning": mode == "frozen_head_training",
        "base_sha256": base_before,
        "check_samples": int(len(check_images)),
    }
    write_json_new(directory / "history.json", history)
    write_json_new(directory / "result.json", result)
    LOGGER.info("Base congelada, salida actualizada y recarga consistentes.")
    return result
