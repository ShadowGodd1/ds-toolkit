"""
ds_toolkit.core.distribution
=============================
DistributionReport: auto-generates histograms, KDE plots, QQ plots,
box plots, and a correlation heatmap for all numeric columns.
Exports as a single self-contained HTML file.

Usage
-----
>>> from ds_toolkit.core import DistributionReport
>>> reporter = DistributionReport()
>>> result = reporter.run(df, output_dir="reports/")
>>> result.display()       # inline in Jupyter
>>> result.html_path       # Path to saved HTML
"""

# Author:  Adnan Mohamud — CEO & Founder, PataDoc (patadoc.com)
# License: MIT

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in notebooks and scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ds_toolkit.base import ReportResult

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

_PALETTE = "#4C72B0"
_GRID_COLOR = "#e8e8e8"
_FONT = "sans-serif"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.color": _GRID_COLOR,
            "grid.linewidth": 0.6,
            "font.family": _FONT,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
        }
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DistributionReport:
    """
    Auto-generates per-column distribution plots and a correlation heatmap.

    Parameters
    ----------
    max_cols : int
        Maximum number of numeric columns to plot individually. Default 30.
    heatmap_method : str
        Correlation method passed to pd.DataFrame.corr(). Default 'pearson'.
    figsize_per_col : tuple
        (width, height) for each individual column figure. Default (10, 3.5).
    """

    def __init__(
        self,
        max_cols: int = 30,
        heatmap_method: str = "pearson",
        figsize_per_col: tuple = (10, 3.5),
    ) -> None:
        self.max_cols = max_cols
        self.heatmap_method = heatmap_method
        self.figsize_per_col = figsize_per_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str | Path] = None,
    ) -> ReportResult:
        """
        Profile the distribution of every numeric column in *df*.

        Parameters
        ----------
        df         : pd.DataFrame
        output_dir : path-like, optional
            Directory to write the HTML report. If None, no file is saved.

        Returns
        -------
        ReportResult
            .figures   — list of matplotlib Figure objects (one per column + heatmap).
            .html_path — Path to saved HTML, or None.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        _style()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return ReportResult(figures=[], html_path=None)

        cols_to_plot = numeric_cols[: self.max_cols]
        figures: List[plt.Figure] = []

        for col in cols_to_plot:
            fig = self._plot_column(df[col])
            figures.append(fig)

        if len(numeric_cols) >= 2:
            heatmap_fig = self._plot_heatmap(df[numeric_cols])
            figures.append(heatmap_fig)

        html_path: Optional[Path] = None
        if output_dir is not None:
            html_path = self._export_html(figures, cols_to_plot, Path(output_dir))

        return ReportResult(figures=figures, html_path=html_path)

    # ------------------------------------------------------------------
    # Per-column plot (hist + KDE, box, QQ — 3 panels)
    # ------------------------------------------------------------------

    def _plot_column(self, series: pd.Series) -> plt.Figure:
        clean = series.dropna()
        col = str(series.name)

        fig, axes = plt.subplots(1, 3, figsize=self.figsize_per_col)
        fig.suptitle(col, fontsize=12, fontweight="bold", x=0.02, ha="left")

        # Panel 1 — Histogram + KDE
        ax = axes[0]
        if len(clean) >= 2:
            ax.hist(
                clean,
                bins="auto",
                color=_PALETTE,
                alpha=0.7,
                density=True,
                edgecolor="white",
                linewidth=0.4,
            )
            try:
                kde = stats.gaussian_kde(clean)
                xs = np.linspace(clean.min(), clean.max(), 300)
                ax.plot(xs, kde(xs), color="#c0392b", linewidth=1.8)
            except Exception:
                pass
        ax.set_title("Histogram + KDE")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        # Panel 2 — Box plot
        ax = axes[1]
        ax.boxplot(
            clean,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=_PALETTE, alpha=0.6),
            medianprops=dict(color="#c0392b", linewidth=2),
            flierprops=dict(marker=".", color="#888", markersize=4),
        )
        ax.set_title("Box Plot")
        ax.set_xticks([])
        ax.set_ylabel(col)

        # Panel 3 — QQ plot
        ax = axes[2]
        if len(clean) >= 4:
            (osm, osr), (slope, intercept, _) = stats.probplot(clean)
            ax.scatter(osm, osr, s=12, color=_PALETTE, alpha=0.6)
            x_line = np.array([min(osm), max(osm)])
            ax.plot(x_line, slope * x_line + intercept, color="#c0392b", linewidth=1.6)
        ax.set_title("Q-Q Plot")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")

        # Annotation: key stats
        if len(clean) > 0:
            stat_text = (
                f"n={len(clean):,}  missing={series.isna().sum():,}\n"
                f"μ={clean.mean():.3g}  σ={clean.std():.3g}\n"
                f"skew={stats.skew(clean):.2f}  kurt={stats.kurtosis(clean):.2f}"
            )
            fig.text(
                0.98,
                0.97,
                stat_text,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#555555",
                transform=fig.transFigure,
                bbox=dict(
                    facecolor="white", alpha=0.7, edgecolor="#dddddd", boxstyle="round,pad=0.3"
                ),
            )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def _plot_heatmap(self, df_numeric: pd.DataFrame) -> plt.Figure:
        n = len(df_numeric.columns)
        size = max(6, min(n * 0.7, 20))
        fig, ax = plt.subplots(figsize=(size, size * 0.8))

        corr = df_numeric.corr(method=self.heatmap_method)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle

        sns.heatmap(
            corr,
            mask=mask,
            annot=n <= 20,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="#f0f0f0",
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title(
            f"Correlation Heatmap ({self.heatmap_method.capitalize()})",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def _export_html(
        self,
        figures: List[plt.Figure],
        col_names: List[str],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "distribution_report.html"

        sections: List[str] = []
        for i, fig in enumerate(figures):
            label = col_names[i] if i < len(col_names) else "Correlation Heatmap"
            b64 = _fig_to_base64(fig)
            sections.append(
                f'<div class="section">'
                f"<h2>{_escape_html(label)}</h2>"
                f'<img src="data:image/png;base64,{b64}" />'
                f"</div>"
            )
        plt.close("all")

        html = _HTML_TEMPLATE.format(
            title="Distribution Report",
            body="\n".join(sections),
        )
        html_path.write_text(html, encoding="utf-8")
        return html_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fa; color: #1a1a2e; margin: 0; padding: 0;
    }}
    header {{
      background: #fff; border-bottom: 1px solid #e5e7eb;
      padding: 18px 32px; position: sticky; top: 0; z-index: 10;
    }}
    header h1 {{ margin: 0; font-size: 17px; font-weight: 600; color: #111; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 24px; }}
    .section {{
      background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
      padding: 20px 24px; margin-bottom: 20px;
    }}
    .section h2 {{
      font-size: 14px; font-weight: 600; color: #374151;
      margin: 0 0 14px 0; padding-bottom: 8px;
      border-bottom: 1px solid #f0f0f0;
    }}
    .section img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  </style>
</head>
<body>
  <header><h1>{title}</h1></header>
  <main>{body}</main>
</body>
</html>"""
