import json
from pathlib import Path
from typing import Protocol

from .domain import Content


class CatalogRepository(Protocol):
    """Replace with a DB repository; return a consistent catalog snapshot per request."""

    def all(self) -> list[Content]: ...


class JsonCatalog:
    def __init__(self, path: Path):
        self.path = path

    def all(self) -> list[Content]:
        items = [Content.model_validate(row) for row in json.loads(self.path.read_text("utf-8"))]
        unique: dict[str, Content] = {}
        for item in items:
            if item.id in unique and unique[item.id] != item:
                raise ValueError(f"Conflicting duplicate catalog id: {item.id}")
            unique[item.id] = item
        return sorted(unique.values(), key=lambda item: item.id)
