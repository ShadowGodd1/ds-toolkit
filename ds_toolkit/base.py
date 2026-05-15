"""
ds_toolkit.base
===============
Shared dataclasses, result containers, and base classes used across all stages.
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Display mixin
# ---------------------------------------------------------------------------


class DisplayMixin:
    """Adds a .display() method that renders the object in Jupyter."""

    def display(self) -> "DisplayMixin":
        """Render rich IPython output. Returns self so calls can be chained."""
        try:
            from IPython.display import display as ipy_display, HTML

            ipy_display(HTML(self._repr_html_()))
        except ImportError:
            print(repr(self))
        return self

    def _repr_html_(self) -> str:  # pragma: no cover
        return f"<pre>{repr(self)}</pre>"


# ---------------------------------------------------------------------------
# Stage 1 result types
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult(DisplayMixin):
    """Output of DataProfiler.profile()."""

    summary_df: pd.DataFrame
    warnings: List[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        warn_html = ""
        if self.warnings:
            items = "".join(f"<li>{w}</li>" for w in self.warnings)
            warn_html = f"""
            <div style="margin-top:12px;padding:10px 14px;background:#fff8e1;
                        border-left:3px solid #f9a825;border-radius:4px;font-size:13px">
              <strong>⚠ Warnings ({len(self.warnings)})</strong>
              <ul style="margin:6px 0 0 16px;padding:0">{items}</ul>
            </div>"""
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:600;margin-bottom:8px">
            DataProfiler — {len(self.summary_df)} columns profiled
          </div>
          {self.summary_df.to_html(border=0, classes='ds-table')}
          {warn_html}
        </div>"""


@dataclass
class ValidationResult(DisplayMixin):
    """Output of SchemaValidator.check()."""

    passed: bool
    violations_df: pd.DataFrame

    def _repr_html_(self) -> str:
        status_color = "#2e7d32" if self.passed else "#c62828"
        status_label = "✓ PASSED" if self.passed else "✗ FAILED"
        body = (
            "<p style='color:#555;margin:8px 0'>No violations found.</p>"
            if self.violations_df.empty
            else self.violations_df.to_html(border=0, index=False)
        )
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <div style="font-weight:700;color:{status_color};font-size:15px;margin-bottom:8px">
            {status_label}
          </div>
          {body}
        </div>"""


@dataclass
class ReportResult(DisplayMixin):
    """Output of DistributionReport.run()."""

    figures: List[Any]
    html_path: Optional[Path] = None

    def _repr_html_(self) -> str:
        path_line = f"<p>Saved to: <code>{self.html_path}</code></p>" if self.html_path else ""
        return f"""
        <div style="font-family:sans-serif;font-size:13px">
          <strong>DistributionReport</strong> — {len(self.figures)} figure(s) generated.
          {path_line}
        </div>"""
