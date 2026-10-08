"""Configuración validada para extracción de características con Xception."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TransferConfig:
    """Configura el Sprint 2, sin reutilizar el clasificador de la Unidad 2.

    image_size: resolución interna; las entradas conservan RGB sin normalizar.
    smoke_test: prueba funcional reducida, nunca evidencia del objetivo >92 %.
    """
    image_size: int = 160
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 0.001
    dropout: float = 0.2
    l2_strength: float = 0.0001
    patience: int = 3
    seed: int = 42
    smoke_test: bool = False
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        """Rechaza tipos, rangos o combinaciones de ejecución inválidas."""
        integer_bounds = (
            ("image_size", 71, 512),
            ("batch_size", 1, 1024),
            ("epochs", 1, 1000),
            ("patience", 0, 1000),
            ("seed", 0, 2**31 - 1),
            ("cpu_threads", 1, 64),
        )
        for name, lower, upper in integer_bounds:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} debe ser entero.")
            if not lower <= value <= upper:
                raise ValueError(f"{name} debe estar entre {lower} y {upper}.")
        for name, lower, upper in (
            ("learning_rate", 0.0, 1.0),
            ("dropout", 0.0, 1.0),
            ("l2_strength", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} debe ser numérico.")
            if not math.isfinite(value):
                raise ValueError(f"{name} debe ser finito.")
            if not lower <= value < upper:
                raise ValueError(f"{name} fuera del rango [{lower}, {upper}).")
        if self.learning_rate == 0:
            raise ValueError("learning_rate debe ser mayor que cero.")
        if type(self.smoke_test) is not bool:
            raise TypeError("smoke_test debe ser booleano.")
        if self.smoke_test and self.epochs != 1:
            raise ValueError("La prueba corta utiliza exactamente una época.")

    def to_dict(self) -> dict[str, object]:
        """Devuelve una copia serializable de esta configuración."""
        return asdict(self)
