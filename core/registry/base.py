from __future__ import annotations

from typing import Dict, Generic, Protocol, Type, TypeVar

T = TypeVar("T")


class ModelRegistry(Generic[T]):
    """Simple registry to map names to model classes."""

    def __init__(self) -> None:
        self._reg: Dict[str, Type[T]] = {}

    def register(self, name: str, cls: Type[T]) -> None:
        self._reg[name] = cls

    def create(self, name: str, **kwargs) -> T:
        if name not in self._reg:
            raise KeyError(name)
        return self._reg[name](**kwargs)

    def names(self) -> list[str]:  # pragma: no cover - convenience
        return sorted(self._reg.keys())


def register(registry: ModelRegistry[T], name: str):
    """Class decorator to register a model."""

    def decorator(cls: Type[T]) -> Type[T]:
        registry.register(name, cls)
        return cls

    return decorator
