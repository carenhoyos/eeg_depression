"""Servicio de informe: niveles, hallazgos frente a referencia, figuras (PNG) y PDF. Parametrizado por Settings.report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

from eegdep.inference import ModelBundle
from eegdep.interpret import topomap_grid

from ..core.config import ReportSection


# ----------------------------------------------------------------- niveles
def assign_tier(p: float, rep: ReportSection) -> dict[str, Any]:
    for t in rep.tiers:
        if t.min <= p < t.max or (p >= 1.0 and t.max >= 1.0):
            return t.model_dump()
    raise ValueError(f"Ningún nivel cubre p={p}; revisa report.tiers")


# ----------------------------------------------------------------- hallazgos
def findings_table(x: pd.Series, z: pd.Series, bundle: ModelBundle, rep: ReportSection) -> list[dict[str, Any]]:
    ref, meta = bundle.reference[rep.reference_group], bundle.feature_meta
    sel = z[z.abs() >= rep.deviation_z_threshold].abs().sort_values(ascending=False).index[: rep.max_findings]
    return [{"feature": f, "kind": meta.loc[f, "kind"], "band": meta.loc[f, "band"], "location": meta.loc[f, "location"],
             "value": float(x[f]), "z": float(z[f]), "reference_p5": float(ref.loc["p5", f]),
             "reference_p50": float(ref.loc["p50", f]), "reference_p95": float(ref.loc["p95", f]),
             "direction": "above" if z[f] > 0 else "below"} for f in sel]


def band_summary(x: pd.Series, bundle: ModelBundle, rep: ReportSection) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    meta, ref = bundle.feature_meta, bundle.reference[rep.reference_group]
    patient, reference = {}, {}
    for b in bundle.pipeline_cfg["bands"]:
        cols = meta.index[(meta.kind == "relpow") & (meta.band == b)]
        if len(cols):
            patient[b] = float(x[cols].mean())
            reference[b] = {p: float(ref.loc[p, cols].mean()) for p in ref.index}
    return patient, reference


def topography_z(z: pd.Series, bundle: ModelBundle, rep: ReportSection) -> dict[str, list[dict[str, float]]]:
    ch = [bundle.pipeline_cfg["channels"]["rename"].get(c, c) for c in bundle.pipeline_cfg["channels"]["list"]]
    out = {}
    for b in rep.topomap_bands:
        names = [f"relpow__{b}__{c}" for c in ch]
        if all(n in z.index for n in names):
            out[b] = [{"channel": c, "value": float(z[n])} for c, n in zip(ch, names)]
    return out


# ----------------------------------------------------------------- figuras
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_probability(res: dict[str, Any], rep: ReportSection, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    for t in rep.tiers:
        ax.barh([0], [t.max - t.min], left=t.min, color=rep.plotting.tier_colors[t.code], alpha=0.25, height=0.6)
        ax.text((t.min + t.max) / 2, 0.45, t.name, ha="center", va="bottom", fontsize=8)
    ax.axvline(res["p_posterior"], color="#222222", lw=2)
    ax.text(res["p_posterior"], -0.5, f"{100 * res['p_posterior']:.0f} %", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.9, 0.9); ax.set_yticks([])
    ax.set_xlabel(f"Probabilidad posterior (prevalencia pre-test {100 * res['pretest_prevalence']:.0f} %)")
    _style(ax); ax.spines["left"].set_visible(False)
    fig.savefig(out, dpi=rep.figures_dpi, bbox_inches="tight"); plt.close(fig)
    return out


def fig_band_power(patient: dict[str, float], reference: dict[str, dict[str, float]], rep: ReportSection, out: Path) -> Path:
    bands, pl = list(patient), rep.plotting
    xs = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    p5, p95 = [reference[b]["p5"] for b in bands], [reference[b]["p95"] for b in bands]
    p25, p75 = [reference[b]["p25"] for b in bands], [reference[b]["p75"] for b in bands]
    ax.bar(xs, np.subtract(p95, p5), bottom=p5, width=0.55, color=pl.color_reference, alpha=0.25, label="Referencia p5–p95")
    ax.bar(xs, np.subtract(p75, p25), bottom=p25, width=0.55, color=pl.color_reference, alpha=0.5, label="Referencia p25–p75")
    ax.plot(xs, [patient[b] for b in bands], "o", ms=8, color=pl.color_primary, label="Paciente", zorder=3)
    ax.set_xticks(xs, bands); ax.set_ylabel("Potencia relativa (media de canales)")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.savefig(out, dpi=rep.figures_dpi, bbox_inches="tight"); plt.close(fig)
    return out


def fig_topomaps(topo: dict[str, list[dict[str, float]]], bundle: ModelBundle, rep: ReportSection, out: Path) -> Path | None:
    if not topo:
        return None
    channels = [d["channel"] for d in next(iter(topo.values()))]
    vals = {b: np.array([d["value"] for d in rows]) for b, rows in topo.items()}
    plot_cfg = {**rep.plotting.model_dump(), "dpi": rep.figures_dpi, "topomap_contours": 4}
    return topomap_grid(vals, channels, bundle.pipeline_cfg["channels"]["montage"], out,
                        "Potencia relativa del paciente: z respecto al grupo de referencia", plot_cfg,
                        diverging=True, center=0.0, cbar_label="z")


def fig_coherence(matrix: np.ndarray, channels: list[str], band: str, rep: ReportSection, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(np.nan_to_num(matrix, nan=0.0), cmap=rep.plotting.sequential_cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(channels)), channels, rotation=90, fontsize=7); ax.set_yticks(range(len(channels)), channels, fontsize=7)
    ax.set_title(f"Coherencia {band} (%)", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=rep.figures_dpi, bbox_inches="tight"); plt.close(fig)
    return out


def fig_contributions(contrib: pd.Series, rep: ReportSection, out: Path) -> Path:
    k, pl = rep.top_contributions, rep.plotting
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:k]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * k + 1))
    ax.barh(top.index, top.values, color=[pl.color_negative if v > 0 else pl.color_primary for v in top.values], height=0.6)
    ax.axvline(0, color=pl.color_reference, lw=1); ax.set_xlabel("Contribución al logit (positivo = hacia depresión)")
    ax.tick_params(axis="y", labelsize=7); _style(ax)
    fig.savefig(out, dpi=rep.figures_dpi, bbox_inches="tight"); plt.close(fig)
    return out


def fig_signal(signal: dict[str, Any], sv, rep: ReportSection, out: Path) -> Path:
    """Trazos apilados del fragmento de señal limpia (uV) y PSD media por canal."""
    data = np.asarray(signal["data"]); t = signal["t0"] + np.arange(data.shape[1]) / signal["sfreq"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    sp = float(sv.trace_spacing_uv)
    for i, (ch, row) in enumerate(zip(signal["channels"], data)):
        ax1.plot(t, row - i * sp, lw=0.6, color=rep.plotting.color_primary)
    ax1.set_yticks([-i * sp for i in range(len(signal["channels"]))], signal["channels"], fontsize=7)
    ax1.set_xlabel("Tiempo (s)"); ax1.set_title("Señal limpia (fragmento)", fontsize=10); _style(ax1)
    psd = np.asarray(signal["psd"]); f = np.asarray(signal["psd_freqs"])
    ax2.plot(f, psd.T, lw=0.6, color=rep.plotting.color_reference, alpha=0.5)
    ax2.plot(f, psd.mean(axis=0), lw=1.8, color=rep.plotting.color_primary, label="Media de canales")
    ax2.set_xlabel("Frecuencia (Hz)"); ax2.set_ylabel("log10 PSD (uV²/Hz)"); ax2.set_title("Espectro", fontsize=10)
    ax2.legend(frameon=False, fontsize=8); _style(ax2)
    fig.savefig(out, dpi=rep.figures_dpi, bbox_inches="tight"); plt.close(fig)
    return out


# ----------------------------------------------------------------- PDF
class _PDF(FPDF):
    def __init__(self, title: str, page_size: str):
        super().__init__(format=page_size)
        self.doc_title = title

    def header(self):
        self.set_font("Helvetica", "B", 11); self.cell(0, 8, self.doc_title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(160, 160, 160); self.line(10, 18, self.w - 10, 18); self.ln(3)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 7)
        self.cell(0, 6, f"Apoyo a la decisión, no diagnóstico. Página {self.page_no()}", align="C")


def _section(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 10); pdf.ln(2); pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9)


def _kv(pdf: FPDF, rows: list[tuple[str, str]]):
    for k, v in rows:
        pdf.set_font("Helvetica", "B", 9); pdf.cell(55, 5.5, k)
        pdf.set_font("Helvetica", "", 9); pdf.multi_cell(0, 5.5, v, new_x="LMARGIN", new_y="NEXT")


def _f(x, d=2):
    return "?" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def build_pdf(report: dict[str, Any], figure_paths: dict[str, Path], rep: ReportSection, out: Path) -> Path:
    """Estructura: identificación, calidad/técnica, resultado e impresión, descripción cuantitativa,
    modelo y validación (TRIPOD+AI), limitaciones, aviso."""
    pdf = _PDF(rep.pdf.title, rep.pdf.page_size)
    pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    rec, q, m = report["recording"], report["quality"], report["model"]
    _section(pdf, "1. Identificación y datos del registro")
    _kv(pdf, [("ID de análisis", report["analysis_id"]), ("Fecha", report["created_at"]),
              ("Referencia del paciente", rec.get("patient_ref") or "-"), ("Edad / sexo", f"{rec.get('age') or '-'} / {rec.get('sex') or '-'}"),
              ("Condición", rec.get("condition") or "no indicada"), ("Medicación", rec.get("medication") or "no indicada"),
              ("Tipo de entrada", q["input_kind"]), ("Registros analizados", str(q["n_records"]))])
    _section(pdf, "2. Calidad y procesamiento")
    for w in q["warnings"] or ["Sin advertencias."]:
        pdf.multi_cell(0, 5, f"- {w}", new_x="LMARGIN", new_y="NEXT")
    for r in report["results"]:
        t, rid = r["tier"], r["record_id"]
        _section(pdf, f"3. Resultado del modelo - registro {rid}")
        _kv(pdf, [("Probabilidad calibrada", f"{100 * r['p_calibrated']:.1f} % (prevalencia de entrenamiento {100 * m['train_prevalence']:.0f} %)"),
                  ("Razón de verosimilitud", _f(r["likelihood_ratio"])), ("Prevalencia pre-test", f"{100 * r['pretest_prevalence']:.0f} %"),
                  ("Probabilidad posterior", f"{100 * r['p_posterior']:.1f} %"), ("Clasificación", f"{t['name']} ({100 * t['min']:.0f}-{100 * t['max']:.0f} %)")])
        pdf.set_font("Helvetica", "B", 9); pdf.multi_cell(0, 5.5, f"Impresión: {t['text']}", new_x="LMARGIN", new_y="NEXT")
        if f"probability_{rid}" in figure_paths:
            pdf.image(str(figure_paths[f"probability_{rid}"]), w=150)
        pdf.set_font("Helvetica", "", 9); pdf.multi_cell(0, 5, r.get("preliminary_assessment", ""), new_x="LMARGIN", new_y="NEXT")
        _section(pdf, "4. Descripción cuantitativa del EEG")
        for name, w in (("signal", 180), ("band_power", 170), ("topomaps", 170), ("coherence", 100)):
            if f"{name}_{rid}" in figure_paths:
                pdf.image(str(figure_paths[f"{name}_{rid}"]), w=w)
        if r["findings"]:
            pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, "Desviaciones destacadas respecto al grupo de referencia:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for f in r["findings"]:
                pdf.multi_cell(0, 4.5, f"- {f['kind']} {f['band']} {f['location']}: {f['value']:.3f} (z={f['z']:+.2f}; ref p5-p95 {f['reference_p5']:.3f}-{f['reference_p95']:.3f})", new_x="LMARGIN", new_y="NEXT")
        if f"contributions_{rid}" in figure_paths:
            pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, "Características que más pesan en la decisión del modelo:", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(figure_paths[f"contributions_{rid}"]), w=160)
    _section(pdf, "5. Modelo y validación (resumen TRIPOD+AI)")
    v = m["validation"]; cv, ho, hoci = v.get("cv_per_repeat_mean", {}), v.get("holdout", {}), v.get("holdout_ci95", {})
    auc_ci = hoci.get("auc", [None, None])
    _kv(pdf, [("Modelo / versión", f"{m['model_name']} / {m['version']}"),
              ("Entrenamiento", f"{m['n_train'].get('cases', '?')} casos, {m['n_train'].get('controls', '?')} controles (un sitio)"),
              ("Validación cruzada por sujeto", f"AUC {_f(cv.get('auc'))}, sensibilidad {_f(cv.get('sensitivity'))}, especificidad {_f(cv.get('specificity'))}"),
              ("Holdout de sujetos", f"AUC {_f(ho.get('auc'))} (IC95 {_f(auc_ci[0])}-{_f(auc_ci[1])})"),
              ("Calibración (holdout)", f"Brier {_f(ho.get('brier'), 3)}, ECE {_f(ho.get('ece'), 3)}")])
    _section(pdf, "6. Limitaciones")
    for l in report["limitations"]:
        pdf.multi_cell(0, 5, f"- {l}", new_x="LMARGIN", new_y="NEXT")
    _section(pdf, "7. Aviso")
    pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4.5, report["disclaimer"], new_x="LMARGIN", new_y="NEXT")
    out.parent.mkdir(parents=True, exist_ok=True); pdf.output(str(out))
    return out
