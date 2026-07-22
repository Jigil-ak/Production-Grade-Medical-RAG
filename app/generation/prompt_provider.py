"""Prompt versioning Protocol and YAML-based implementation.

Protocol and implementation co-located — no centralized protocols.py.
Reads YAML files from /app/config/prompts/, one per purpose.
PROMPT_VERSION setting selects the active prompt without code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from app.core.logging import get_logger
from app.core.types import PromptTemplate

logger = get_logger(__name__)


@runtime_checkable
class PromptProvider(Protocol):
    """Interface for prompt template providers."""

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Get a PromptTemplate by name and optional version string."""
        ...


class YAMLPromptProvider:
    """Prompt template provider loading from YAML files in app/config/prompts/."""

    def __init__(self, prompts_dir: str = "./app/config/prompts") -> None:
        """Initialize provider and load YAML templates.

        Args:
            prompts_dir: Directory containing YAML prompt template files.
        """
        self._prompts_dir = Path(prompts_dir)
        self._templates: dict[str, PromptTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all YAML prompt files from prompts directory."""
        if not self._prompts_dir.exists():
            logger.warn("Prompts directory does not exist", path=str(self._prompts_dir))
            return

        yaml_files = list(self._prompts_dir.glob("*.yaml")) + list(self._prompts_dir.glob("*.yml"))
        for yfile in yaml_files:
            try:
                data = yaml.safe_load(yfile.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    template = PromptTemplate(
                        name=str(data.get("name", "")),
                        version=str(data.get("version", "")),
                        system_prompt=str(data.get("system_prompt", "")),
                        user_template=str(data.get("user_template", "")),
                        description=str(data.get("description", "")),
                        created_date=str(data.get("created_date", "")),
                    )
                    # Register by version key and name key
                    self._templates[template.version] = template
                    self._templates[template.name] = template
            except Exception as e:
                logger.error("Failed to load prompt template YAML", file=yfile.name, error=str(e))

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Get prompt template matching version or name.

        Args:
            name: Template name or version identifier.
            version: Optional explicit version identifier.

        Returns:
            PromptTemplate model.
        """
        lookup_key = version or name
        if lookup_key in self._templates:
            return self._templates[lookup_key]

        if name in self._templates:
            return self._templates[name]

        raise ValueError(f"Prompt template '{lookup_key}' not found in {self._prompts_dir}")
