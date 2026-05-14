"""
ds_toolkit.infra.config
=========================
ConfigManager: YAML/JSON config loader with ${ENV_VAR} override syntax.

Features
--------
  • Loads YAML or JSON config files
  • Resolves ${ENV_VAR} placeholders in string values
  • Dot-access on the returned Config object (cfg.model.n_estimators)
  • Version-stamps each config on load (load_timestamp, toolkit_version)
  • Validates required keys against a schema dict

Usage
-----
>>> from ds_toolkit.infra import ConfigManager
>>> cfg = ConfigManager.load("config/experiment.yaml")
>>> cfg.model.n_estimators
>>> cfg.data.target_col
>>>
>>> # With required key validation
>>> cfg = ConfigManager.load(
...     "experiment.yaml",
...     required=["data.target_col", "model.task"],
... )
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ds_toolkit


class Config:
    """
    Dot-accessible config object.

    Wraps a nested dict so values can be read as cfg.section.key
    instead of cfg["section"]["key"].
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain nested dict."""
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a value with a default."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        keys = list(self.__dict__.keys())
        return f"Config({keys})"


class ConfigManager:
    """
    Static config loader — no instantiation required.
    """

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        required: Optional[List[str]] = None,
        env_prefix: Optional[str] = None,
    ) -> Config:
        """
        Load a YAML or JSON config file.

        Parameters
        ----------
        path     : str or Path — config file path
        required : list, optional — dot-notation keys that must be present,
                   e.g. ["data.target_col", "model.task"]
        env_prefix : str, optional — prefix for environment variable overrides.
                   If set, env vars like {PREFIX}_MODEL_N_ESTIMATORS override
                   nested config values.

        Returns
        -------
        Config — dot-accessible config object with version stamp.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = cls._read_file(path)
        raw = cls._resolve_env_vars(raw)

        if env_prefix:
            raw = cls._apply_env_overrides(raw, env_prefix)

        # Version stamp
        raw.setdefault("__meta__", {})
        raw["__meta__"]["load_timestamp"] = datetime.utcnow().isoformat()
        raw["__meta__"]["toolkit_version"] = ds_toolkit.__version__
        raw["__meta__"]["config_path"]     = str(path.resolve())

        cfg = Config(raw)

        if required:
            cls._validate_required(cfg, required, path)

        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create a Config object directly from a dict."""
        return Config(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        text   = path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(text) or {}
            except ImportError as exc:
                raise ImportError(
                    "YAML config requires PyYAML: pip install pyyaml"
                ) from exc

        if suffix == ".json":
            return json.loads(text)

        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json."
        )

    @classmethod
    def _resolve_env_vars(cls, obj: Any) -> Any:
        """Recursively resolve ${VAR_NAME} placeholders."""
        if isinstance(obj, str):
            return re.sub(
                r"\$\{([^}]+)\}",
                lambda m: os.environ.get(m.group(1), m.group(0)),
                obj,
            )
        if isinstance(obj, dict):
            return {k: cls._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._resolve_env_vars(v) for v in obj]
        return obj

    @classmethod
    def _apply_env_overrides(cls, data: Dict, prefix: str) -> Dict:
        """
        Apply environment variable overrides.
        E.g. DS_MODEL_N_ESTIMATORS=500 overrides data["model"]["n_estimators"].
        """
        prefix_upper = prefix.upper() + "_"
        for key, val in os.environ.items():
            if not key.upper().startswith(prefix_upper):
                continue
            parts = key[len(prefix_upper):].lower().split("_")
            cls._set_nested(data, parts, cls._coerce(val))
        return data

    @staticmethod
    def _set_nested(d: Dict, keys: List[str], value: Any) -> None:
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    @staticmethod
    def _coerce(val: str) -> Any:
        """Try to coerce string env var value to int / float / bool."""
        if val.lower() in ("true", "yes"):
            return True
        if val.lower() in ("false", "no"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    @staticmethod
    def _validate_required(cfg: Config, required: List[str], path: Path) -> None:
        missing = []
        for dotkey in required:
            parts = dotkey.split(".")
            obj   = cfg
            found = True
            for part in parts:
                if not hasattr(obj, part):
                    found = False
                    break
                obj = getattr(obj, part)
            if not found:
                missing.append(dotkey)
        if missing:
            raise KeyError(
                f"Config '{path}' missing required keys: {missing}"
            )
