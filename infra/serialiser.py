"""
ds_toolkit.infra.serialiser
=============================
PipelineSerialiser: versioned model serialisation with SHA-256 integrity check.

What it saves
-------------
  • The fitted sklearn pipeline / estimator (pickled)
  • A sidecar .json metadata file containing:
      - SHA-256 checksum of the .pkl
      - toolkit version
      - save timestamp
      - estimator class name
      - any user-supplied metadata dict

On load, the checksum is recomputed and compared. Raises on mismatch
(configurable via checksum_on_load).

Usage
-----
>>> from ds_toolkit.infra import PipelineSerialiser
>>> serial = PipelineSerialiser()
>>>
>>> # Save
>>> result = serial.save(pipeline, name="rf_v1", metadata={"roc_auc": 0.91})
>>> result.path        # Path to .pkl
>>> result.checksum    # SHA-256 hex string
>>>
>>> # Load
>>> pipeline = serial.load(result.path)
"""
# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ds_toolkit


@dataclass
class SaveResult:
    """Output of PipelineSerialiser.save()."""
    path: Path
    checksum: str
    metadata_path: Path


class ChecksumError(Exception):
    """Raised when a loaded .pkl fails SHA-256 verification."""


class PipelineSerialiser:
    """
    Versioned model serialisation with integrity verification.

    Parameters
    ----------
    output_dir : str or Path
        Default directory for saved models. Default './models'.
    checksum_on_load : bool
        Raise ChecksumError if SHA-256 mismatches on load. Default True.
    protocol : int
        Pickle protocol version. Default highest available.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./models",
        checksum_on_load: bool = True,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        self.output_dir      = Path(output_dir)
        self.checksum_on_load = checksum_on_load
        self.protocol        = protocol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        pipeline,
        name: str,
        output_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SaveResult:
        """
        Serialise *pipeline* to a versioned .pkl file.

        Parameters
        ----------
        pipeline   : fitted estimator or sklearn Pipeline
        name       : base name — timestamp is appended automatically
        output_dir : override default output directory
        metadata   : optional dict stored in sidecar .json

        Returns
        -------
        SaveResult
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pkl_path  = out_dir / f"{name}_{timestamp}.pkl"
        meta_path = out_dir / f"{name}_{timestamp}.meta.json"

        # Pickle
        pkl_bytes = pickle.dumps(pipeline, protocol=self.protocol)
        pkl_path.write_bytes(pkl_bytes)

        # Checksum
        checksum = hashlib.sha256(pkl_bytes).hexdigest()

        # Metadata sidecar
        meta = {
            "name":            name,
            "timestamp":       timestamp,
            "toolkit_version": ds_toolkit.__version__,
            "estimator_class": type(pipeline).__name__,
            "sha256":          checksum,
            "pkl_path":        str(pkl_path),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return SaveResult(path=pkl_path, checksum=checksum, metadata_path=meta_path)

    def load(self, path: Union[str, Path]) -> Any:
        """
        Load and verify a .pkl saved by this serialiser.

        Parameters
        ----------
        path : str or Path — path to the .pkl file

        Returns
        -------
        The deserialised object.

        Raises
        ------
        FileNotFoundError   — if .pkl or sidecar .json is missing
        ChecksumError       — if SHA-256 mismatches and checksum_on_load=True
        """
        pkl_path  = Path(path)
        meta_path = pkl_path.with_suffix("").with_suffix(".meta.json")

        if not pkl_path.exists():
            raise FileNotFoundError(f"Model file not found: {pkl_path}")

        pkl_bytes = pkl_path.read_bytes()
        actual_checksum = hashlib.sha256(pkl_bytes).hexdigest()

        # Checksum verification
        if self.checksum_on_load and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = meta.get("sha256", "")
            if expected and actual_checksum != expected:
                raise ChecksumError(
                    f"SHA-256 mismatch for '{pkl_path}'.\n"
                    f"  Expected : {expected}\n"
                    f"  Actual   : {actual_checksum}\n"
                    "The file may have been tampered with or corrupted."
                )

        return pickle.loads(pkl_bytes)

    def load_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load the sidecar metadata JSON without deserialising the model."""
        pkl_path  = Path(path)
        meta_path = pkl_path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))
