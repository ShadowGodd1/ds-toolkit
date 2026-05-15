"""
ds_toolkit.reporting
=====================
Stage 7 — Reporting & Notebook Output.

NotebookReporter — renders experiment summary inline in Jupyter
HTMLExporter     — single self-contained HTML export
ModelCard        — structured model card (Mitchell et al. format)
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ds_toolkit.base import DisplayMixin

# ===========================================================================
# NotebookReporter
# ===========================================================================


class NotebookReporter:
    """
    Renders a complete experiment summary inline in Jupyter.

    Displays: metric cards, ranked model table, top feature importance,
    best model diagnostics.

    Usage
    -----
    >>> from ds_toolkit.reporting import NotebookReporter
    >>> reporter = NotebookReporter()
    >>> reporter.display(cv_results, eval_results, shap_result)
    """

    def display(
        self,
        cv_results=None,
        eval_results=None,
        shap_result=None,
        title: str = "Experiment Summary",
    ) -> None:
        """Render inline in Jupyter. Returns None."""
        try:
            from IPython.display import display as ipy_display, HTML

            html = self._build_html(cv_results, eval_results, shap_result, title)
            ipy_display(HTML(html))
        except ImportError:
            print("[NotebookReporter] IPython not available — printing summary.")
            self._print_summary(cv_results, eval_results)

    # ------------------------------------------------------------------

    def _build_html(self, cv, ev, shap, title) -> str:
        parts = [f"""
        <div style="font-family:sans-serif;max-width:900px">
          <h2 style="font-size:18px;font-weight:600;margin:0 0 16px">{title}</h2>
        """]

        # Metric cards from eval_results
        if ev is not None:
            cards_html = ""
            for metric, row in ev.metrics_df.iterrows():
                val = row["value"]
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                cards_html += f"""
                <div style="display:inline-block;background:#f8f9fa;
                            border:0.5px solid #e0e0e0;border-radius:6px;
                            padding:10px 18px;margin:0 8px 8px 0;min-width:110px">
                  <div style="font-size:11px;color:#666;margin-bottom:4px">{metric}</div>
                  <div style="font-size:20px;font-weight:600;color:#1a1a1a">{val_str}</div>
                </div>"""
            parts.append(f"""
            <div style="margin-bottom:20px">
              <div style="font-size:13px;font-weight:600;margin-bottom:8px">Metrics</div>
              {cards_html}
            </div>""")

        # CV ranked model table
        if cv is not None:
            parts.append(f"""
            <div style="margin-bottom:20px">
              <div style="font-size:13px;font-weight:600;margin-bottom:8px">
                Model Ranking — {cv.scoring}
              </div>
              {cv.summary_df.to_html(border=0, float_format=lambda x: f"{x:.4f}",
                                      classes="ds-table", index=False)}
            </div>""")

        # SHAP bar chart
        if shap is not None and "bar" in shap.figures:
            b64 = _fig_to_b64(shap.figures["bar"])
            if b64:
                parts.append(f"""
                <div style="margin-bottom:20px">
                  <div style="font-size:13px;font-weight:600;margin-bottom:8px">
                    Feature Importance (SHAP)
                  </div>
                  <img src="data:image/png;base64,{b64}"
                       style="max-width:600px;border:0.5px solid #e0e0e0;border-radius:4px"/>
                </div>""")

        parts.append("</div>")
        return "".join(parts)

    def _print_summary(self, cv, ev) -> None:
        if ev is not None:
            print("\n── Metrics ──")
            print(ev.metrics_df.to_string())
        if cv is not None:
            print("\n── Model Ranking ──")
            print(cv.summary_df.to_string(index=False))


# ===========================================================================
# HTMLExporter
# ===========================================================================


@dataclass
class ExportResult:
    """Output of HTMLExporter.export()."""

    html_path: Path


class HTMLExporter:
    """
    Assembles results into a single self-contained HTML file.

    All images are base64-embedded; no external CSS or JS required.
    Suitable for emailing or sharing without a Jupyter environment.

    Usage
    -----
    >>> exporter = HTMLExporter()
    >>> result = exporter.export(
    ...     output_path="report.html",
    ...     cv_results=cv_results,
    ...     eval_results=eval_results,
    ...     shap_result=shap_result,
    ...     diagnostic_result=diag_result,
    ... )
    >>> result.html_path
    """

    def export(
        self,
        output_path: Union[str, Path],
        cv_results=None,
        eval_results=None,
        shap_result=None,
        diagnostic_result=None,
        title: str = "DS Toolkit — Experiment Report",
    ) -> ExportResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_full_html(
            title, cv_results, eval_results, shap_result, diagnostic_result
        )
        output_path.write_text(html, encoding="utf-8")
        return ExportResult(html_path=output_path)

    # ------------------------------------------------------------------

    def _build_full_html(self, title, cv, ev, shap, diag) -> str:
        body_parts: List[str] = []

        # Metrics table
        if ev is not None:
            body_parts.append(f"<h2>Metrics</h2>{ev.metrics_df.to_html(border=0)}")

        # CV summary
        if cv is not None:
            body_parts.append(
                f"<h2>Model Ranking — {cv.scoring}</h2>"
                f"{cv.summary_df.to_html(border=0, index=False, float_format=lambda x: f'{x:.4f}')}"
            )

        # SHAP figures
        if shap is not None:
            body_parts.append("<h2>SHAP Importance</h2>")
            for name, fig in shap.figures.items():
                b64 = _fig_to_b64(fig)
                if b64:
                    body_parts.append(
                        f'<p style="font-size:12px;color:#666">{name}</p>'
                        f'<img src="data:image/png;base64,{b64}" style="max-width:700px"/><br/>'
                    )

        # Diagnostic figures
        if diag is not None:
            body_parts.append("<h2>Diagnostics</h2>")
            for name, fig in diag.figures.items():
                b64 = _fig_to_b64(fig)
                if b64:
                    body_parts.append(
                        f'<p style="font-size:12px;color:#666">{name}</p>'
                        f'<img src="data:image/png;base64,{b64}" style="max-width:700px"/><br/>'
                    )

        body = "\n".join(body_parts) or "<p>No results provided.</p>"
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 960px; margin: 40px auto; padding: 0 20px;
            font-size: 13px; color: #1a1a1a; }}
    h1   {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
    h2   {{ font-size: 15px; font-weight: 600; margin: 28px 0 10px;
            padding-bottom: 4px; border-bottom: 1px solid #e5e5e5; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    td, th {{ padding: 6px 10px; border-bottom: 1px solid #f0f0f0;
              text-align: left; }}
    th   {{ font-weight: 600; background: #fafafa; }}
    img  {{ display: block; margin: 8px 0; border: 0.5px solid #ddd;
            border-radius: 4px; }}
    .meta {{ font-size: 11px; color: #888; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated {ts} · ds_toolkit</p>
  {body}
</body>
</html>"""


# ===========================================================================
# ModelCard
# ===========================================================================


@dataclass
class ModelCard(DisplayMixin):
    """
    Structured model card following the Mitchell et al. format.

    Sections
    --------
    model_overview   — estimator class, task type
    training_data    — from ProfileResult if provided
    performance      — from MetricsResult and CVResults
    feature_importance — from ShapResult
    limitations      — from ErrorReport
    experiment_log   — run_id, git hash, config path
    """

    model_name: str
    task: str
    metrics: Optional[Dict[str, float]] = None
    cv_summary: Optional[pd.DataFrame] = None
    top_features: Optional[List[Dict]] = None
    data_summary: Optional[Dict] = None
    limitations: Optional[List[str]] = None
    experiment_info: Optional[Dict] = None
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    )

    def _repr_html_(self) -> str:
        sections: List[str] = []

        def section(title: str, content: str) -> str:
            return (
                f'<div style="margin-bottom:20px">'
                f'<div style="font-size:13px;font-weight:600;border-bottom:1px solid #e5e5e5;'
                f'padding-bottom:4px;margin-bottom:8px">{title}</div>'
                f"{content}</div>"
            )

        # Overview
        sections.append(
            section(
                "Model Overview",
                f"""
        <table style="border-collapse:collapse;font-size:12px">
          <tr><td style="padding:3px 12px 3px 0;color:#666">Model</td>
              <td style="padding:3px 0"><code>{self.model_name}</code></td></tr>
          <tr><td style="padding:3px 12px 3px 0;color:#666">Task</td>
              <td style="padding:3px 0">{self.task.upper()}</td></tr>
          <tr><td style="padding:3px 12px 3px 0;color:#666">Generated</td>
              <td style="padding:3px 0">{self.generated_at}</td></tr>
        </table>""",
            )
        )

        # Performance
        if self.metrics:
            rows = "".join(
                f"<tr><td style='padding:3px 12px 3px 0;color:#666'>{k}</td>"
                f"<td style='padding:3px 0'><strong>{v:.4f}</strong></td></tr>"
                for k, v in self.metrics.items()
            )
            sections.append(
                section(
                    "Performance Metrics",
                    f"""
            <table style="border-collapse:collapse;font-size:12px">{rows}</table>""",
                )
            )

        # CV summary
        if self.cv_summary is not None:
            sections.append(
                section(
                    "Cross-Validation Results",
                    self.cv_summary.to_html(
                        border=0, index=False, float_format=lambda x: f"{x:.4f}"
                    ),
                )
            )

        # Feature importance
        if self.top_features:
            feat_rows = "".join(
                f"<tr><td style='padding:3px 12px 3px 0'>{f['feature']}</td>"
                f"<td style='padding:3px 0'>{f['importance']:.4f}</td></tr>"
                for f in self.top_features
            )
            sections.append(
                section(
                    "Top Features (SHAP)",
                    f"""
            <table style="border-collapse:collapse;font-size:12px">
              <tr style="font-weight:500"><td>Feature</td><td>Mean |SHAP|</td></tr>
              {feat_rows}
            </table>""",
                )
            )

        # Limitations
        if self.limitations:
            items = "".join(f"<li>{lim}</li>" for lim in self.limitations)
            sections.append(section("Known Limitations", f"<ul>{items}</ul>"))

        # Experiment info
        if self.experiment_info:
            exp_rows = "".join(
                f"<tr><td style='padding:3px 12px 3px 0;color:#666'>{k}</td>"
                f"<td style='padding:3px 0'><code>{v}</code></td></tr>"
                for k, v in self.experiment_info.items()
            )
            sections.append(
                section(
                    "Experiment Log",
                    f"""
            <table style="border-collapse:collapse;font-size:12px">{exp_rows}</table>""",
                )
            )

        return '<div style="font-family:sans-serif;max-width:700px">' + "".join(sections) + "</div>"

    def to_html(self) -> str:
        """Return the full HTML string."""
        return self._repr_html_()

    def to_md(self) -> str:
        """Return a Markdown string of the model card."""
        lines = [
            f"# Model Card — {self.model_name}",
            "",
            f"**Task:** {self.task.upper()}",
            f"**Generated:** {self.generated_at}",
            "",
        ]
        if self.metrics:
            lines += ["## Performance Metrics", ""]
            lines += [f"| {k} | {v:.4f} |" for k, v in self.metrics.items()]
            lines.append("")
        if self.limitations:
            lines += ["## Known Limitations", ""]
            lines += [f"- {lim}" for lim in self.limitations]
        return "\n".join(lines)


def generate_model_card(
    model,
    cv_results=None,
    eval_results=None,
    shap_result=None,
    error_report=None,
    experiment_info: Optional[Dict] = None,
) -> ModelCard:
    """
    Convenience function to build a ModelCard from toolkit result objects.

    Parameters
    ----------
    model         : fitted estimator
    cv_results    : CVResults, optional
    eval_results  : MetricsResult, optional
    shap_result   : ShapResult, optional
    error_report  : ErrorReport, optional
    experiment_info : dict, optional

    Returns
    -------
    ModelCard
    """
    model_name = type(model).__name__
    task = cv_results.task if cv_results else eval_results.task if eval_results else "unknown"

    metrics = None
    if eval_results is not None:
        metrics = eval_results.metrics_df["value"].to_dict()

    cv_summary = cv_results.summary_df if cv_results else None

    top_features = None
    if shap_result is not None:
        mean_abs = np.abs(shap_result.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][: shap_result.top_n]
        top_features = [
            {"feature": shap_result.feature_names[i], "importance": float(mean_abs[i])}
            for i in top_idx
        ]

    limitations = None
    if error_report is not None:
        flagged = error_report.segments_df[error_report.segments_df["flagged"]]
        if not flagged.empty:
            limitations = [
                f"High error shift on feature '{f}' (std_shift={row['std_shift']:.3f})"
                for f, row in flagged.iterrows()
            ]

    return ModelCard(
        model_name=model_name,
        task=task,
        metrics=metrics,
        cv_summary=cv_summary,
        top_features=top_features,
        limitations=limitations,
        experiment_info=experiment_info,
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _fig_to_b64(fig) -> Optional[str]:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None
