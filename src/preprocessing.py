"""Validación y preprocesamiento para el modelo final Xception."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

class ImageValidationError(ValueError):
    """Error controlado de validación de imagen."""

def validate_uploaded_image(
    content: bytes,
    filename: str,
    allowed_extensions: tuple[str, ...],
    max_bytes: int,
) -> Image.Image:
    if not filename:
        raise ImageValidationError("El archivo no tiene nombre.")
    if Path(filename).suffix.lower() not in allowed_extensions:
        raise ImageValidationError("Formato no permitido.")
    if not content:
        raise ImageValidationError("El archivo está vacío.")
    if len(content) > max_bytes:
        raise ImageValidationError("El archivo supera el límite permitido.")
    try:
        with Image.open(BytesIO(content)) as probe:
            probe.verify()
        with Image.open(BytesIO(content)) as image:
            result = image.convert("RGB")
            result.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("El contenido no es una imagen válida.") from exc
    return result

def prepare_image(
    image: Image.Image,
    target_size: tuple[int, int] = (160, 160),
) -> np.ndarray:
    image = image.convert("RGB")
    resized = image.resize(target_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    expected = (target_size[1], target_size[0], 3)
    if array.shape != expected:
        raise ImageValidationError(f"Forma inesperada: {array.shape}.")
    return np.expand_dims(array, axis=0)
