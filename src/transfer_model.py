"""Xception congelado y nueva salida CIFAR-10 mediante capas estándar de Keras."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .transfer_config import TransferConfig

if TYPE_CHECKING:
    import keras


def build_transfer_classifier(
    config: TransferConfig, *, weights: str | None = "imagenet"
) -> tuple[keras.Model, keras.Model]:
    """Construye el clasificador y devuelve también su base congelada.

    weights=None se permite únicamente para pruebas de arquitectura sin descarga.
    El ejecutable de entrenamiento siempre solicita weights='imagenet'.
    """
    import keras

    if weights not in ("imagenet", None):
        raise ValueError("Solo se admiten pesos ImageNet o None para pruebas.")
    if keras.backend.image_data_format() != "channels_last":
        raise ValueError("Configura Keras con image_data_format='channels_last'.")
    base = keras.applications.Xception(
        include_top=False,
        weights=weights,
        input_shape=(config.image_size, config.image_size, 3),
    )
    base.trainable = False
    # Las imágenes se reciben RGB, rango 0..255; no deben normalizarse fuera.
    inputs = keras.Input(shape=(None, None, 3), dtype="float32", name="image_rgb")
    x = keras.layers.Resizing(
        config.image_size, config.image_size,
        interpolation="bilinear", name="resize_xception"
    )(inputs)
    x = keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1.0, name="scale_xception"
    )(x)
    # Mantiene BatchNormalization en inferencia, incluso en futuro fine-tuning.
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = keras.layers.Dropout(
        config.dropout, seed=config.seed, name="head_dropout"
    )(x)
    outputs = keras.layers.Dense(
        10, activation="softmax",
        kernel_regularizer=keras.regularizers.L2(config.l2_strength),
        name="class_probabilities"
    )(x)
    model = keras.Model(inputs, outputs, name="eikonorasis_akrivis")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    if base.trainable_weights:
        raise RuntimeError("La base Xception no quedó congelada.")
    return model, base


def weights_digest(weights: Iterable[object]) -> str:
    """Calcula una huella SHA-256 de pesos para comprobar que no cambiaron."""
    import keras

    digest = hashlib.sha256()
    for weight in weights:
        array = np.asarray(keras.ops.convert_to_numpy(weight))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parameter_counts(model: keras.Model) -> dict[str, int]:
    """Cuenta parámetros totales y entrenables, sin contar el optimizador."""
    trainable = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def validate_probabilities(probabilities: np.ndarray, sample_count: int) -> None:
    """Comprueba diez probabilidades finitas por imagen y suma unitaria."""
    values = np.asarray(probabilities)
    if values.shape != (sample_count, 10):
        raise ValueError(f"Salida inesperada: {values.shape}, se esperaba ({sample_count}, 10).")
    if not np.isfinite(values).all():
        raise ValueError("La salida contiene NaN o infinito.")
    if (values < -1e-6).any() or (values > 1.0 + 1e-6).any():
        raise ValueError("Hay probabilidades fuera de [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Las probabilidades no suman uno por imagen.")
