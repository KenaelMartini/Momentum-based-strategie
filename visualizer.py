# ============================================================
# visualizer.py — Dashboard de Visualisation Complet
# ============================================================
# RÔLE DE CE FICHIER :
# Générer 3 formats de visualisation pour la stratégie momentum :
#   1. Matplotlib -> PNG + PDF haute qualite (rapports)
#   2. Plotly     -> HTML interactif (presentation live)
#   3. Streamlit  -> Dashboard live (monitoring ghost test)
#
# USAGE :
#   python visualizer.py                     # genere les 3 formats
#   python visualizer.py --format matplotlib # PNG + PDF seulement
#   python visualizer.py --format plotly     # HTML seulement
#   python visualizer.py --format streamlit  # genere dashboard_live.py
#
# DEPENDANCES :
#   pip install matplotlib plotly streamlit kaleido pandas numpy
# ============================================================

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Imports matplotlib au niveau module ────────────────────
# On importe ici pour que toutes les methodes _mpl_* y aient acces
try:
    import matplotlib
    matplotlib.use("Agg")   # Backend sans fenetre - obligatoire en script
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ── Imports plotly au niveau module ────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import INITIAL_CAPITAL, RISK_FREE_RATE


# ============================================================
# CLASSE PRINCIPALE : StrategyVisualizer
# ============================================================

class StrategyVisualizer:
    """
    Visualiseur complet de la strategie momentum.

    Charge tous les resultats produits par les phases precedentes
    et genere des visualisations professionnelles en 3 formats.
    """

    def __init__(
        self,
        results_dir     : str   = "./results",
        output_dir      : str   = "./results/charts",
        initial_capital : float = INITIAL_CAPITAL,
    ):
        self.results_dir     = Path(results_dir)
        self.output_dir      = Path(output_dir)
        self.initial_capital = initial_capital
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Donnees chargees
        self.returns      = None
        self.mc_data      = None
        self.yearly_stats = None

        # Valeurs du portefeuille
        self.portfolio_values = None
        self.benchmark_values = None
        self.drawdown_series  = None

        # Metriques — TOUTES stockees en float des le chargement
        self.metrics = {}

        logger.info(f"StrategyVisualizer initialise | Output : {self.output_dir}")

    # ─────────────────────────────────────────────────────────
    # CHARGEMENT DES DONNEES
    # ─────────────────────────────────────────────────────────
    def load_data(self) -> bool:
        """
        Charge toutes les donnees depuis les fichiers de resultats.

        Cherche dans ./results/validation/ les fichiers produits
        par validator.py. Fallback sur ./results/backtest_*/.

        Returns:
            True si les donnees sont chargees avec succes
        """
        logger.info("Chargement des donnees...")

        val_dir = self.results_dir / "validation"
        if val_dir.exists():
            returns_files = sorted(val_dir.glob("returns_*.csv"))
            mc_files      = sorted(val_dir.glob("monte_carlo_*.csv"))
            summary_files = sorted(val_dir.glob("summary_*.csv"))

            if returns_files:
                df           = pd.read_csv(returns_files[-1], index_col=0, parse_dates=True)
                self.returns = df.iloc[:, 0]
                logger.info(f"  Rendements charges : {len(self.returns)} jours")

            if mc_files:
                self.mc_data = pd.read_csv(mc_files[-1])
                logger.info(f"  Monte Carlo charge : {len(self.mc_data)} simulations")

            if summary_files:
                summary = pd.read_csv(summary_files[-1], index_col=0)
                raw     = summary.iloc[:, 0].to_dict()

                # CONVERSION SYSTEMATIQUE EN FLOAT
                # Les CSV stockent tout en string — on force le cast ici
                # pour ne jamais avoir d'erreur "format code f for str" plus bas
                for k, v in raw.items():
                    try:
                        self.metrics[k] = float(v)
                    except (ValueError, TypeError):
                        self.metrics[k] = v   # garde tel quel si non convertible

                logger.info(f"  Metriques chargees : {len(self.metrics)} valeurs")

        # Fallback : donnees backtest si pas de validation
        if self.returns is None:
            bt_dirs = sorted(self.results_dir.glob("backtest_*"))
            if bt_dirs:
                bt_dir   = bt_dirs[-1]
                ret_file = bt_dir / "returns.csv"
                if ret_file.exists():
                    df           = pd.read_csv(ret_file, index_col=0, parse_dates=True)
                    col          = "strategy_returns" if "strategy_returns" in df else df.columns[0]
                    self.returns = df[col]
                    logger.info(f"  Rendements backtest charges : {len(self.returns)} jours")

        if self.returns is None:
            logger.error("Aucune donnee trouvee. Lance run_backtest.py et validator.py d'abord.")
            return False

        self.returns = self.returns.dropna()
        self._compute_portfolio_values()
        self._compute_yearly_stats()

        return True

    def _safe_metric(self, key: str, default: float = 0.0) -> float:
        """
        Recupere une metrique en float de facon robuste.

        Methode utilitaire utilisee dans tous les panels pour
        eviter les erreurs de type string/float.

        Args:
            key     : cle dans self.metrics
            default : valeur par defaut si cle absente

        Returns:
            float garanti
        """
        val = self.metrics.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _compute_portfolio_values(self):
        """Calcule les valeurs du portefeuille, benchmark et drawdown."""
        simple                = np.exp(self.returns) - 1
        self.portfolio_values = self.initial_capital * (1 + simple).cumprod()

        # Benchmark proxy : rendement constant tres faible
        # Serie pandas avec le meme index que les returns
        bm_daily              = pd.Series(0.00025, index=self.returns.index)
        self.benchmark_values = self.initial_capital * (1 + bm_daily).cumprod()

        # Drawdown
        peak                  = self.portfolio_values.expanding().max()
        self.drawdown_series  = (self.portfolio_values - peak) / peak

    def _compute_yearly_stats(self):
        """Calcule les statistiques par annee."""
        rf_daily = (1 + RISK_FREE_RATE) ** (1/252) - 1
        stats    = {}

        for year in sorted(self.returns.index.year.unique()):
            yr = self.returns[self.returns.index.year == year].dropna()
            if len(yr) < 50:
                continue

            pv     = self.initial_capital * (1 + (np.exp(yr) - 1)).cumprod()
            cagr   = (pv.iloc[-1] / self.initial_capital) ** (252 / len(yr)) - 1
            sharpe = (yr.mean() - rf_daily) / yr.std() * np.sqrt(252) if yr.std() > 0 else 0.0
            peak   = pv.expanding().max()
            max_dd = float(((pv - peak) / peak).min())

            stats[year] = {"cagr": float(cagr), "sharpe": float(sharpe), "max_dd": max_dd}

        self.yearly_stats = stats

    # ─────────────────────────────────────────────────────────
    # FORMAT 1 : MATPLOTLIB (PNG + PDF)
    # ─────────────────────────────────────────────────────────
    def export_matplotlib(self, dpi: int = 150) -> list:
        """
        Genere le dashboard complet en PNG et PDF.

        MATPLOTLIB — LE STANDARD POUR LES RAPPORTS :
        Rendu parfait haute resolution, reproductible, compatible
        Word/LaTeX/PowerPoint. Pas de dependances JavaScript.

        STRUCTURE 3x3 :
          [Equity Curve (2 cols)]   [Metriques cles]
          [Drawdown (2 cols)]       [Rolling Sharpe]
          [Monte Carlo]  [Annuel]   [Heatmap phases]

        Returns:
            liste des chemins [png_path, pdf_path]
        """
        if not MATPLOTLIB_OK:
            logger.error("matplotlib non installe. pip install matplotlib")
            return []

        logger.info("\n  Generation Matplotlib...")

        # Style dark professionnel
        plt.rcParams.update({
            "figure.facecolor" : "#0d1117",
            "axes.facecolor"   : "#161b22",
            "axes.edgecolor"   : "#30363d",
            "axes.labelcolor"  : "#c9d1d9",
            "text.color"       : "#c9d1d9",
            "xtick.color"      : "#8b949e",
            "ytick.color"      : "#8b949e",
            "grid.color"       : "#21262d",
            "grid.alpha"       : 0.8,
            "font.family"      : "monospace",
            "axes.spines.top"  : False,
            "axes.spines.right": False,
        })

        fig = plt.figure(figsize=(22, 16), facecolor="#0d1117")
        fig.suptitle(
            "MOMENTUM STRATEGY -- PERFORMANCE DASHBOARD",
            fontsize=18, fontweight="bold", color="#58a6ff",
            y=0.98, fontfamily="monospace"
        )

        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            hspace=0.45, wspace=0.35,
            left=0.06, right=0.97,
            top=0.93, bottom=0.06
        )

        ax1 = fig.add_subplot(gs[0, :2])
        self._mpl_equity_curve(ax1)

        ax2 = fig.add_subplot(gs[0, 2])
        self._mpl_metrics_table(ax2)

        ax3 = fig.add_subplot(gs[1, :2])
        self._mpl_drawdown(ax3)

        ax4 = fig.add_subplot(gs[1, 2])
        self._mpl_rolling_sharpe(ax4)

        ax5 = fig.add_subplot(gs[2, 0])
        self._mpl_monte_carlo(ax5)

        ax6 = fig.add_subplot(gs[2, 1])
        self._mpl_yearly_performance(ax6)

        ax7 = fig.add_subplot(gs[2, 2])
        self._mpl_metrics_heatmap(ax7)

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = self.output_dir / f"dashboard_{ts}.png"
        pdf_path = self.output_dir / f"dashboard_{ts}.pdf"

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="#0d1117")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

        logger.info(f"  PNG sauvegarde : {png_path}")
        logger.info(f"  PDF sauvegarde : {pdf_path}")
        return [str(png_path), str(pdf_path)]

    # ── Panels Matplotlib ─────────────────────────────────────

    def _mpl_equity_curve(self, ax):
        """Equity curve avec separation IS/OOS."""
        dates      = self.portfolio_values.index
        values     = self.portfolio_values.values
        bm_values  = self.benchmark_values.values
        split_date = pd.Timestamp(self.metrics.get("oos_start", "2022-01-01"))

        is_mask  = dates < split_date
        oos_mask = dates >= split_date

        ax.plot(dates[is_mask],  values[is_mask],  color="#58a6ff", lw=1.8, label="Strategie IS",  zorder=3)
        ax.plot(dates[oos_mask], values[oos_mask], color="#3fb950", lw=1.8, label="Strategie OOS", zorder=3)
        ax.plot(dates, bm_values, color="#8b949e", lw=1.0, alpha=0.5, label="Benchmark", zorder=2)

        ax.axvspan(split_date, dates[-1], alpha=0.05, color="#3fb950", zorder=1)
        ax.axvline(split_date, color="#3fb950", lw=1.0, ls="--", alpha=0.6)
        ax.text(split_date, values.max() * 0.95, "OOS->", color="#3fb950", fontsize=8, ha="left")

        ax.fill_between(dates[is_mask],  self.initial_capital, values[is_mask],  alpha=0.08, color="#58a6ff")
        ax.fill_between(dates[oos_mask], self.initial_capital, values[oos_mask], alpha=0.08, color="#3fb950")

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.set_title("EQUITY CURVE", color="#58a6ff", fontsize=10, fontweight="bold", pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#161b22")

    def _mpl_drawdown(self, ax):
        """Drawdown depuis le dernier sommet."""
        dd    = self.drawdown_series
        dates = dd.index

        ax.fill_between(dates, dd.values, 0, color="#f85149", alpha=0.7)
        ax.plot(dates, dd.values, color="#f85149", lw=0.8)

        max_dd_date = dd.idxmin()
        ax.annotate(
            f"Max DD: {dd.min():.1%}",
            xy=(max_dd_date, float(dd.min())),
            xytext=(max_dd_date, float(dd.min()) * 0.5),
            color="#f85149", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#f85149", lw=0.8)
        )

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title("DRAWDOWN", color="#f85149", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylim(float(dd.min()) * 1.3, 0.02)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#161b22")

    def _mpl_rolling_sharpe(self, ax):
        """Rolling Sharpe 252 jours."""
        rf_daily    = (1 + RISK_FREE_RATE) ** (1/252) - 1
        roll_mean   = self.returns.rolling(252).mean()
        roll_std    = self.returns.rolling(252).std()
        roll_sharpe = (roll_mean - rf_daily) / roll_std * np.sqrt(252)

        rs_vals = roll_sharpe.values
        rs_idx  = roll_sharpe.index

        ax.plot(rs_idx, rs_vals, color="#d2a8ff", lw=1.5)
        ax.fill_between(rs_idx, rs_vals, 0, where=rs_vals > 0, color="#3fb950", alpha=0.2)
        ax.fill_between(rs_idx, rs_vals, 0, where=rs_vals < 0, color="#f85149", alpha=0.2)
        ax.axhline(0, color="#8b949e", lw=0.8, ls="--")

        rs_clean = roll_sharpe.dropna()
        mean_rs  = float(rs_clean.mean())
        ax.axhline(mean_rs, color="#58a6ff", lw=0.8, ls=":", alpha=0.7,
                label=f"Moy: {mean_rs:.2f}")

        ax.set_title("ROLLING SHARPE (252j)", color="#d2a8ff", fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#161b22")

    def _mpl_monte_carlo(self, ax):
        """Distribution Monte Carlo des Sharpe simules."""
        if self.mc_data is None or "sharpes" not in self.mc_data.columns:
            ax.text(0.5, 0.5, "Monte Carlo\nnon disponible",
                    ha="center", va="center", color="#8b949e", transform=ax.transAxes)
            ax.set_facecolor("#161b22")
            return

        sharpes  = self.mc_data["sharpes"].dropna().values
        observed = self._safe_metric("is_sharpe", float(np.mean(sharpes)))
        p5       = float(np.percentile(sharpes, 5))

        n, bins, patches = ax.hist(sharpes, bins=50, color="#58a6ff", alpha=0.7, edgecolor="none")

        for patch, left in zip(patches, bins[:-1]):
            if left < p5:
                patch.set_facecolor("#f85149")
                patch.set_alpha(0.8)

        ax.axvline(observed,          color="#3fb950", lw=2.0, label=f"Observe: {observed:.3f}")
        ax.axvline(p5,                color="#f85149", lw=1.5, ls="--", label=f"5%ile: {p5:.3f}")
        ax.axvline(float(np.median(sharpes)), color="#d2a8ff", lw=1.5, ls=":",
                   label=f"Mediane: {np.median(sharpes):.3f}")

        ax.set_title("MONTE CARLO SHARPE", color="#58a6ff", fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=7, framealpha=0.3)
        ax.set_xlabel("Sharpe Ratio", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#161b22")

    def _mpl_yearly_performance(self, ax):
        """Bar chart des CAGR annuels."""
        if not self.yearly_stats:
            ax.set_facecolor("#161b22")
            return

        years  = sorted(self.yearly_stats.keys())
        cagrs  = [self.yearly_stats[y]["cagr"] for y in years]
        colors = ["#3fb950" if c > 0 else "#f85149" for c in cagrs]

        bars = ax.bar(years, [c * 100 for c in cagrs], color=colors, alpha=0.8, width=0.6)

        for bar, cagr in zip(bars, cagrs):
            h    = bar.get_height()
            ypos = h + 0.5 if cagr > 0 else h - 2.5
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                f"{cagr:.0%}", ha="center",
                va="bottom" if cagr > 0 else "top",
                fontsize=7, color="#c9d1d9"
            )

        avg_cagr = float(np.mean(cagrs)) * 100
        ax.axhline(avg_cagr, color="#58a6ff", lw=1.2, ls="--", label=f"Moy: {avg_cagr:.1f}%")
        ax.axhline(0, color="#8b949e", lw=0.8)

        ax.set_title("CAGR ANNUEL", color="#3fb950", fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#161b22")

    def _mpl_metrics_table(self, ax):
        """Tableau des metriques cles IS vs OOS."""
        ax.set_facecolor("#161b22")
        ax.axis("off")

        # _safe_metric garantit des floats — plus d'erreur format
        metrics_display = [
            ("IS CAGR",      self._safe_metric("is_cagr"),            ".2%", "#58a6ff"),
            ("OOS CAGR",     self._safe_metric("oos_cagr"),           ".2%", "#3fb950"),
            ("IS Sharpe",    self._safe_metric("is_sharpe"),          ".3f", "#58a6ff"),
            ("OOS Sharpe",   self._safe_metric("oos_sharpe"),         ".3f", "#3fb950"),
            ("IS Max DD",    self._safe_metric("is_max_dd"),          ".2%", "#f85149"),
            ("OOS Max DD",   self._safe_metric("oos_max_dd"),         ".2%", "#f85149"),
            ("IS/OOS Ratio", self._safe_metric("sharpe_ratio"),       ".2f", "#d2a8ff"),
            ("% Yrs Pos",    self._safe_metric("pct_positive_years"), ".1%", "#3fb950"),
            ("MC Sharpe 5%", self._safe_metric("mc_sharpe_5pct"),     ".3f", "#d2a8ff"),
        ]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(metrics_display) + 1)
        ax.set_title("KEY METRICS", color="#58a6ff", fontsize=10, fontweight="bold", pad=8)

        for i, (label, value, fmt, color) in enumerate(reversed(metrics_display)):
            y = i + 0.5
            ax.text(0.05, y, label, color="#8b949e", fontsize=9, va="center")
            ax.text(0.95, y, format(value, fmt), color=color,
                    fontsize=10, fontweight="bold", va="center", ha="right")
            ax.axhline(y + 0.5, color="#21262d", lw=0.5, alpha=0.5)

    def _mpl_metrics_heatmap(self, ax):
        """Heatmap de comparaison des phases de developpement."""
        phases        = ["Baseline", "Phase 2A", "Phase 2B"]
        metrics_names = ["CAGR", "Sharpe", "Max DD", "Turnover"]

        # Valeurs absolues hardcodees (resultats obtenus)
        data = np.array([
            [7.13,  0.145, -34.81, 76.9],
            [13.17, 0.575, -22.46, 93.2],
            [12.80, 0.643, -12.47,  5.3],
        ])

        # Normalisation 0->1 par colonne
        # Pour Max DD et Turnover : moins = mieux -> on inverse
        data_abs = np.abs(data)  # Max DD devient positif : 34.81, 22.46, 12.47

        norm = np.zeros_like(data_abs, dtype=float)
        for j in range(data_abs.shape[1]):
            col_min = data_abs[:, j].min()
            col_max = data_abs[:, j].max()
            rng     = col_max - col_min + 1e-8
            if j in [2, 3]:  # Max DD et Turnover : valeur absolue plus petite = mieux = VERT
                norm[:, j] = 1 - (data_abs[:, j] - col_min) / rng
            else:
                norm[:, j] = (data_abs[:, j] - col_min) / rng

        ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(metrics_names)))
        ax.set_xticklabels(metrics_names, fontsize=8, color="#c9d1d9")
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases, fontsize=8, color="#c9d1d9")

        labels = [
            ["7.1%",   "0.145", "-34.8%", "76.9%"],
            ["13.2%",  "0.575", "-22.5%", "93.2%"],
            ["12.8%",  "0.643", "-12.5%",  "5.3%"],
        ]
        for i in range(len(phases)):
            for j in range(len(metrics_names)):
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

        ax.set_title("PHASES COMPARISON", color="#d2a8ff", fontsize=10, fontweight="bold", pad=8)
        ax.set_facecolor("#161b22")

    # ─────────────────────────────────────────────────────────
    # FORMAT 2 : PLOTLY (HTML interactif)
    # ─────────────────────────────────────────────────────────
    def export_plotly(self) -> str:
        """
        Genere le dashboard interactif HTML avec Plotly.

        PLOTLY — LE STANDARD POUR LES PRESENTATIONS :
        Interactif (zoom, hover), HTML autonome, qualite Bloomberg.

        Returns:
            chemin du fichier HTML genere
        """
        if not PLOTLY_OK:
            logger.error("plotly non installe. pip install plotly")
            return ""

        logger.info("\n  Generation Plotly HTML...")

        COLORS = {
            "bg"    : "#0d1117", "paper": "#161b22", "grid": "#21262d",
            "text"  : "#c9d1d9", "blue" : "#58a6ff", "green": "#3fb950",
            "red"   : "#f85149", "purple": "#d2a8ff", "gray": "#8b949e",
        }

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Equity Curve -- IS vs OOS vs Benchmark",
                "Drawdown",
                "Rolling Sharpe (252j)",
                "Monte Carlo Distribution",
                "Performance Annuelle",
                "Comparaison des Phases",
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"},     {"type": "heatmap"}],
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        # ── Equity Curve ───────────────────────────────────────
        split_date = pd.Timestamp(self.metrics.get("oos_start", "2022-01-01"))
        pv         = self.portfolio_values
        is_pv      = pv[pv.index < split_date]
        oos_pv     = pv[pv.index >= split_date]

        fig.add_trace(go.Scatter(
            x=is_pv.index, y=is_pv.values, name="IS",
            line=dict(color=COLORS["blue"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>IS</extra>"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=oos_pv.index, y=oos_pv.values, name="OOS",
            line=dict(color=COLORS["green"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>OOS</extra>"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.benchmark_values.index, y=self.benchmark_values.values,
            name="Benchmark", line=dict(color=COLORS["gray"], width=1, dash="dot"), opacity=0.6,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>BM</extra>"
        ), row=1, col=1)

        fig.add_vrect(x0=split_date, x1=pv.index[-1],
                      fillcolor=COLORS["green"], opacity=0.05,
                      layer="below", line_width=0, row=1, col=1)
        fig.add_vline(x=split_date, line_dash="dash",
                      line_color=COLORS["green"], opacity=0.5, row=1, col=1)

        # ── Drawdown ───────────────────────────────────────────
        dd = self.drawdown_series
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, name="Drawdown",
            fill="tozeroy", line=dict(color=COLORS["red"], width=1),
            fillcolor="rgba(248,81,73,0.3)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>DD</extra>"
        ), row=2, col=1)

        # ── Monte Carlo ────────────────────────────────────────
        if self.mc_data is not None and "sharpes" in self.mc_data.columns:
            sharpes  = self.mc_data["sharpes"].dropna()
            observed = self._safe_metric("is_sharpe", float(sharpes.mean()))
            p5       = float(np.percentile(sharpes, 5))

            fig.add_trace(go.Histogram(
                x=sharpes, nbinsx=50, name="MC Sharpe",
                marker_color=COLORS["blue"], opacity=0.7,
                hovertemplate="Sharpe: %{x:.3f}<br>Count: %{y}<extra></extra>"
            ), row=2, col=2)

            fig.add_vline(x=float(observed), line_color=COLORS["green"], line_width=2,
                         annotation_text=f"Observe: {float(observed):.3f}",
                         annotation_font_color=COLORS["green"], row=2, col=2)
            fig.add_vline(x=p5, line_color=COLORS["red"], line_width=1.5, line_dash="dash",
                         annotation_text=f"5pct: {p5:.3f}",
                         annotation_font_color=COLORS["red"], row=2, col=2)

        # ── Performance annuelle ───────────────────────────────
        if self.yearly_stats:
            years  = sorted(self.yearly_stats.keys())
            cagrs  = [self.yearly_stats[y]["cagr"] * 100 for y in years]
            colors = [COLORS["green"] if c > 0 else COLORS["red"] for c in cagrs]

            fig.add_trace(go.Bar(
                x=years, y=cagrs, name="CAGR Annuel",
                marker_color=colors, opacity=0.85,
                text=[f"{c:.1f}%" for c in cagrs], textposition="outside",
                hovertemplate="%{x}<br>CAGR: %{y:.2f}%<extra></extra>"
            ), row=3, col=1)

            fig.add_hline(y=float(np.mean(cagrs)), line_dash="dot",
                         line_color=COLORS["blue"],
                         annotation_text=f"Moy: {float(np.mean(cagrs)):.1f}%",
                         annotation_font_color=COLORS["blue"], row=3, col=1)

        # ── Heatmap des phases ─────────────────────────────────
        phases        = ["Baseline", "Phase 2A", "Phase 2B"]
        metrics_names = ["CAGR%", "Sharpe", "|MaxDD|%", "Turnover%"]
        text_values   = [
            ["7.1%",  "0.145", "-34.8%", "76.9%"],
            ["13.2%", "0.575", "-22.5%", "93.2%"],
            ["12.8%", "0.643", "-12.5%",  "5.3%"],
        ]
        z_norm = [
            [0.54, 0.00, 0.00, 0.82],
            [1.00, 0.85, 0.55, 0.06],
            [0.97, 1.00, 1.00, 1.00],
        ]

        fig.add_trace(go.Heatmap(
            z=z_norm, x=metrics_names, y=phases,
            text=text_values,
            colorscale="RdYlGn",
            showscale=False,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"
        ), row=3, col=2)

        # Annotations texte manuelles sur la heatmap
        for i, phase in enumerate(phases):
            for j, metric in enumerate(metrics_names):
                fig.add_annotation(
                    x=metric, y=phase,
                    text=text_values[i][j],
                    font=dict(color="white", size=11, family="monospace"),
                    showarrow=False,
                    row=3, col=2
        )

        # ── Layout global ──────────────────────────────────────
        fig.update_layout(
            title=dict(
                text    ="MOMENTUM STRATEGY -- INTERACTIVE DASHBOARD",
                font    =dict(size=20, color=COLORS["blue"], family="monospace"),
                x=0.5, xanchor="center",
            ),
            paper_bgcolor = COLORS["bg"],
            plot_bgcolor  = COLORS["paper"],
            font          = dict(color=COLORS["text"], family="monospace"),
            height        = 1100,
            showlegend    = True,
            legend        = dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=COLORS["grid"], borderwidth=1),
            hoverlabel    = dict(bgcolor=COLORS["paper"], bordercolor=COLORS["blue"]),
        )

        for r in range(1, 4):
            for c in range(1, 3):
                try:
                    fig.update_xaxes(gridcolor=COLORS["grid"], row=r, col=c)
                    fig.update_yaxes(gridcolor=COLORS["grid"], row=r, col=c)
                except Exception:
                    pass

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"dashboard_{ts}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={"displayModeBar": True, "scrollZoom": True}
        )

        logger.info(f"  HTML sauvegarde : {html_path}")
        return str(html_path)

    # ─────────────────────────────────────────────────────────
    # FORMAT 3 : STREAMLIT (Dashboard live)
    # ─────────────────────────────────────────────────────────
    def export_dashboard(self) -> str:
        """
        Genere le script Streamlit pour le dashboard live.

        STREAMLIT — IDEAL POUR LE GHOST TEST :
        Se met a jour automatiquement (TTL 30s), interface web locale,
        tout en Python. Lance avec : streamlit run dashboard_live.py

        Returns:
            chemin du script dashboard_live.py genere
        """
        logger.info("\n  Generation du dashboard Streamlit...")

        code = r'''# ============================================================
# dashboard_live.py — Dashboard Live Streamlit
# ============================================================
# USAGE : streamlit run dashboard_live.py
# Se rafraichit automatiquement toutes les 30 secondes.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import time, sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Momentum Strategy Dashboard",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; font-family: monospace; }
    .metric-label { font-size: 12px; color: #8b949e; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    h1, h2, h3 { color: #58a6ff !important; font-family: monospace !important; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR     = Path("./results")
INITIAL_CAPITAL = 100_000
RISK_FREE_RATE  = 0.05
COLORS = {
    "bg": "#0d1117", "paper": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "blue": "#58a6ff", "green": "#3fb950",
    "red": "#f85149", "purple": "#d2a8ff", "gray": "#8b949e",
}

@st.cache_data(ttl=30)
def load_returns():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("returns_*.csv"))
        if files:
            df = pd.read_csv(files[-1], index_col=0, parse_dates=True)
            return df.iloc[:, 0].dropna()
    bt_dirs = sorted(RESULTS_DIR.glob("backtest_*"))
    if bt_dirs:
        f = bt_dirs[-1] / "returns.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            col = "strategy_returns" if "strategy_returns" in df else df.columns[0]
            return df[col].dropna()
    return None

@st.cache_data(ttl=30)
def load_mc():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("monte_carlo_*.csv"))
        if files:
            return pd.read_csv(files[-1])
    return None

@st.cache_data(ttl=30)
def load_metrics():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("summary_*.csv"))
        if files:
            df = pd.read_csv(files[-1], index_col=0)
            raw = df.iloc[:, 0].to_dict()
            result = {}
            for k, v in raw.items():
                try:
                    result[k] = float(v)
                except Exception:
                    result[k] = v
            return result
    return {}

def compute_portfolio(returns):
    simple = np.exp(returns) - 1
    pv     = INITIAL_CAPITAL * (1 + simple).cumprod()
    peak   = pv.expanding().max()
    dd     = (pv - peak) / peak
    return pv, dd

def safe_metric(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except Exception:
        return default

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    oos_split    = st.date_input("OOS Split Date", value=datetime(2022, 1, 1))
    show_bm      = st.checkbox("Show Benchmark", value=True)
    st.divider()
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── CHARGEMENT ─────────────────────────────────────────────
returns = load_returns()
mc_data = load_mc()
metrics = load_metrics()

if returns is None:
    st.error("Aucune donnee trouvee. Lance run_backtest.py et validator.py d\'abord.")
    st.stop()

pv, dd   = compute_portfolio(returns)
oos_date = pd.Timestamp(oos_split)
rf       = (1 + RISK_FREE_RATE) ** (1/252) - 1

cagr   = float((pv.iloc[-1] / INITIAL_CAPITAL) ** (252/len(returns)) - 1)
sharpe = float((returns.mean() - rf) / returns.std() * np.sqrt(252))
max_dd = float(dd.min())
vol    = float(returns.std() * np.sqrt(252))

# ── HEADER ─────────────────────────────────────────────────
st.title("MOMENTUM STRATEGY DASHBOARD")
st.caption(f"Periode : {returns.index[0].date()} -> {returns.index[-1].date()} | Capital : ${INITIAL_CAPITAL:,.0f}")
st.divider()

# ── METRIQUES ──────────────────────────────────────────────
cols = st.columns(6)
display = [
    ("CAGR",       cagr,   ".1%", cagr > 0),
    ("Sharpe",     sharpe, ".3f", sharpe > 0.5),
    ("Max DD",     max_dd, ".1%", max_dd > -0.20),
    ("Volatilite", vol,    ".1%", True),
    ("OOS CAGR",   safe_metric(metrics, "oos_cagr"),   ".1%", True),
    ("OOS Sharpe", safe_metric(metrics, "oos_sharpe"), ".3f", True),
]
for col, (label, value, fmt, good) in zip(cols, display):
    css = "positive" if good else "negative"
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {css}">{format(value, fmt)}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── EQUITY CURVE + DRAWDOWN ────────────────────────────────
fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.03,
    subplot_titles=["Equity Curve", "Drawdown"]
)

is_pv  = pv[pv.index < oos_date]
oos_pv = pv[pv.index >= oos_date]

fig_main.add_trace(go.Scatter(
    x=is_pv.index, y=is_pv.values, name="IS",
    line=dict(color=COLORS["blue"], width=2),
    hovertemplate="%{x|%Y-%m-%d} $%{y:,.0f}<extra>IS</extra>"
), row=1, col=1)

fig_main.add_trace(go.Scatter(
    x=oos_pv.index, y=oos_pv.values, name="OOS",
    line=dict(color=COLORS["green"], width=2),
    hovertemplate="%{x|%Y-%m-%d} $%{y:,.0f}<extra>OOS</extra>"
), row=1, col=1)

if show_bm:
    bm = INITIAL_CAPITAL * (1 + pd.Series(0.00025, index=returns.index)).cumprod()
    fig_main.add_trace(go.Scatter(
        x=bm.index, y=bm.values, name="Benchmark",
        line=dict(color=COLORS["gray"], width=1, dash="dot"), opacity=0.5
    ), row=1, col=1)

fig_main.add_trace(go.Scatter(
    x=dd.index, y=dd.values * 100, name="Drawdown",
    fill="tozeroy", line=dict(color=COLORS["red"], width=1),
    fillcolor="rgba(248,81,73,0.3)"
), row=2, col=1)

fig_main.add_vrect(x0=oos_date, x1=pv.index[-1],
                   fillcolor=COLORS["green"], opacity=0.04, layer="below", line_width=0)
fig_main.update_layout(
    height=600, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
    font=dict(color=COLORS["text"], family="monospace"),
    legend=dict(bgcolor="rgba(22,27,34,0.8)"), hovermode="x unified"
)
fig_main.update_xaxes(gridcolor=COLORS["grid"])
fig_main.update_yaxes(gridcolor=COLORS["grid"])
fig_main.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
fig_main.update_yaxes(ticksuffix="%", row=2, col=1)
st.plotly_chart(fig_main, use_container_width=True)

# ── MONTE CARLO + ANNUEL ───────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Monte Carlo Distribution")
    if mc_data is not None and "sharpes" in mc_data.columns:
        sharpes  = mc_data["sharpes"].dropna()
        observed = sharpe
        p5       = float(np.percentile(sharpes, 5))
        fig_mc   = go.Figure()
        fig_mc.add_trace(go.Histogram(x=sharpes, nbinsx=50, marker_color=COLORS["blue"], opacity=0.7))
        fig_mc.add_vline(x=float(observed), line_color=COLORS["green"], line_width=2,
                        annotation_text=f"Obs: {float(observed):.3f}", annotation_font_color=COLORS["green"])
        fig_mc.add_vline(x=p5, line_color=COLORS["red"], line_dash="dash",
                        annotation_text=f"5pct: {p5:.3f}", annotation_font_color=COLORS["red"])
        fig_mc.update_layout(height=300, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                             font=dict(color=COLORS["text"]), showlegend=False)
        st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Lance validator.py pour les donnees Monte Carlo")

with col_r:
    st.subheader("Performance Annuelle")
    yearly = {}
    for year in returns.index.year.unique():
        yr = returns[returns.index.year == year].dropna()
        if len(yr) < 50:
            continue
        pvy  = INITIAL_CAPITAL * (1 + (np.exp(yr) - 1)).cumprod()
        c    = float((pvy.iloc[-1] / INITIAL_CAPITAL) ** (252/len(yr)) - 1)
        yearly[year] = c

    years  = sorted(yearly.keys())
    cagrs  = [yearly[y] * 100 for y in years]
    colors = [COLORS["green"] if c > 0 else COLORS["red"] for c in cagrs]
    fig_yr = go.Figure(go.Bar(
        x=years, y=cagrs, marker_color=colors,
        text=[f"{c:.1f}%" for c in cagrs], textposition="outside", opacity=0.85
    ))
    fig_yr.update_layout(height=300, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                         font=dict(color=COLORS["text"]), showlegend=False)
    fig_yr.update_xaxes(gridcolor=COLORS["grid"])
    fig_yr.update_yaxes(gridcolor=COLORS["grid"], ticksuffix="%")
    st.plotly_chart(fig_yr, use_container_width=True)

# ── ROLLING SHARPE ─────────────────────────────────────────
st.subheader("Rolling Sharpe (252j)")
roll_s = (returns.rolling(252).mean() - rf) / returns.rolling(252).std() * np.sqrt(252)
fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=roll_s.index, y=roll_s.values, name="Rolling Sharpe",
    line=dict(color=COLORS["purple"], width=1.5), fill="tozeroy"
))
fig_rs.add_hline(y=0,   line_color=COLORS["gray"],  line_width=0.8)
fig_rs.add_hline(y=0.5, line_color=COLORS["green"], line_dash="dot",
                annotation_text="Seuil 0.5", annotation_font_color=COLORS["green"])
fig_rs.update_layout(height=250, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                     font=dict(color=COLORS["text"]), showlegend=False)
fig_rs.update_xaxes(gridcolor=COLORS["grid"])
fig_rs.update_yaxes(gridcolor=COLORS["grid"])
st.plotly_chart(fig_rs, use_container_width=True)

st.divider()
st.caption(f"Mise a jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Strategie Momentum Phase 2D")

if auto_refresh:
    time.sleep(30)
    st.rerun()
'''

        path = Path("./dashboard_live.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"  Dashboard Streamlit sauvegarde : {path}")
        logger.info("  Lance avec : streamlit run dashboard_live.py")
        return str(path)

    # ─────────────────────────────────────────────────────────
    # METHODE PRINCIPALE : genere les 3 formats
    # ─────────────────────────────────────────────────────────
    def generate_all(self) -> dict:
        """Genere les 3 formats en une seule commande."""
        logger.info("\n" + "="*55)
        logger.info("  GENERATION DES VISUALISATIONS")
        logger.info("="*55)

        if not self.load_data():
            logger.error("Impossible de charger les donnees.")
            return {}

        results = {}

        logger.info("\n  [1/3] Matplotlib PNG + PDF...")
        results["matplotlib"] = self.export_matplotlib()

        logger.info("\n  [2/3] Plotly HTML interactif...")
        results["plotly"] = self.export_plotly()

        logger.info("\n  [3/3] Streamlit dashboard live...")
        results["streamlit"] = self.export_dashboard()

        logger.info("\n" + "="*55)
        logger.info("  FICHIERS GENERES")
        logger.info("="*55)
        logger.info(f"  PNG/PDF   : {self.output_dir}/dashboard_*.png/pdf")
        logger.info(f"  HTML      : {self.output_dir}/dashboard_*.html")
        logger.info(f"  Streamlit : ./dashboard_live.py")
        logger.info("  Pour lancer le dashboard : streamlit run dashboard_live.py")

        return results


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Visualisation de la strategie momentum")
    parser.add_argument("--format", choices=["all", "matplotlib", "plotly", "streamlit"],
                        default="all", help="Format de sortie")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir",  default="./results/charts")
    args = parser.parse_args()

    print("=" * 60)
    print("  VISUALISATION -- STRATEGIE MOMENTUM")
    print("=" * 60)

    viz = StrategyVisualizer(
        results_dir     = args.results_dir,
        output_dir      = args.output_dir,
        initial_capital = INITIAL_CAPITAL,
    )

    if args.format == "all":
        results = viz.generate_all()
    else:
        if not viz.load_data():
            sys.exit(1)
        if args.format == "matplotlib":
            results = {"matplotlib": viz.export_matplotlib()}
        elif args.format == "plotly":
            results = {"plotly": viz.export_plotly()}
        elif args.format == "streamlit":
            results = {"streamlit": viz.export_dashboard()}
        else:
            results = {}

    print("\n" + "=" * 60)
    print("  FICHIERS GENERES")
    print("=" * 60)
    for fmt, paths in results.items():
        if isinstance(paths, list):
            for p in paths:
                if p:
                    print(f"  {fmt:12s} : {p}")
        elif paths:
            print(f"  {fmt:12s} : {paths}")

    if results.get("streamlit"):
        print("\n  Pour lancer le dashboard live :")
        print("  streamlit run dashboard_live.py")