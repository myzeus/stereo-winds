#!/usr/bin/env python
"""Schematic of the three-stage training pipeline for cross-satellite stereo winds.

(a) Teacher fine-tuning: a WindFlow-RAFT + differentiable-WLS teacher is fine-tuned
    from radiosondes (Huber on u,v) + a self-supervised chi-squared geometric residual,
    anchored to a frozen pretrained checkpoint (KL on height).
(b) Offline label generation: the frozen teacher runs once over the whole matched
    GEO–GEO corpus; its outputs (u, v, h, sigma, chi2) and a QA weight are cached to disk.
(c) Student distillation: a single-viewpoint student is distilled from the stored labels
    via heteroscedastic Gaussian NLLs; it predicts its own per-pixel uncertainty (teacher
    sigma is NOT distilled).

Flat schematic — no data is loaded. Renders instantly:

    python scripts/fig_training_overview.py

writes figures/fig_training_overview.{png,pdf}. Style matches fig_methodology_architecture.py.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fig_training_overview")
REPO = Path(__file__).resolve().parent.parent

# --- palette (extends fig_methodology_architecture.py) ----------------------
COLOR_A = "#e8743b"        # teacher / GOES-16 — orange
COLOR_B = "#19a7ce"        # GOES-18 — blue
COLOR_STUDENT = "#2a9d8f"  # student — teal
COLOR_LOSS = "#e76f51"     # loss / supervision — coral
COLOR_FROZEN = "#9aa0a6"   # frozen elements — gray
COLOR_DATA = "#e8e8ea"     # stored-data glyphs — light gray
COLOR_BOX = "#f4f4f4"      # stage-box fill
COLOR_BOX_EDGE = "#333333"
COLOR_ARROW = "#444444"

FIGW, FIGH = 12.0, 8.4     # compressed footprint; original was (14.0, 12.0)
ASP = FIGW / FIGH          # fig-x fraction -> fig-y fraction for a square

TILE_CMAP = {"u": "RdBu_r", "v": "RdBu_r", "h": "viridis",
             "sig": "cividis", "chi2": "magma"}
TILE_DIVERGING = {"u", "v"}
TILE_ITEMS = [("u", "$u$"), ("v", "$v$"), ("h", "$h$"),
              ("sig", r"$\sigma_{h,u,v}$"), ("chi2", r"$\chi^2$")]
TILE_CENTERS = [0.310, 0.341, 0.372, 0.403, 0.434]   # 5 output tiles

# teacher / frozen-teacher block geometry — IDENTICAL in bands (a) and (b)
TEACHER_RECT = (0.200, -0.075, 0.098, 0.150)   # (x[unused], dy-from-center, w, h)


# ---------------------------------------------------------------------------
# Style helpers (mirrored from fig_methodology_architecture.py)
# ---------------------------------------------------------------------------
def _stage_box(fig, rect, label, edge=COLOR_BOX_EDGE, lw=1.1, label_color="#111"):
    x, y, w, h = rect
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor=COLOR_BOX, edgecolor=edge,
        linewidth=lw, zorder=1, clip_on=False)
    fig.patches.append(box)
    if label:
        fig.text(x + 0.010, y + h - 0.010, label, ha="left", va="top",
                 fontsize=9, fontweight="bold", color=label_color, zorder=5)


def _down_arrow(fig, x, y0, y1, color=COLOR_ARROW, lw=1.5, label=None,
                label_side="right"):
    fig.patches.append(FancyArrowPatch(
        (x, y0), (x, y1), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=13, lw=lw, color=color, zorder=4, clip_on=False))
    if label:
        dx = 0.012 if label_side == "right" else -0.012
        ha = "left" if label_side == "right" else "right"
        fig.text(x + dx, (y0 + y1) / 2, label, ha=ha, va="center",
                 fontsize=7.2, style="italic", color=color, zorder=5)


def _arrow(fig, xy0, xy1, color=COLOR_ARROW, lw=1.3, rad=0.0, ls="-"):
    """General (optionally curved) arrow in figure coords."""
    fig.patches.append(FancyArrowPatch(
        xy0, xy1, transform=fig.transFigure, arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}", mutation_scale=12, lw=lw,
        color=color, linestyle=ls, zorder=4, clip_on=False))


# ---------------------------------------------------------------------------
# Schematic glyph helpers
# ---------------------------------------------------------------------------
def draw_padlock(fig, x, y, color, w=0.012):
    """Tiny closed padlock: rounded body + semicircular shackle on top."""
    bw, bh = w, w * ASP * 0.62
    fig.patches.append(FancyBboxPatch(
        (x - bw / 2, y - bh / 2), bw, bh,
        boxstyle="round,pad=0,rounding_size=0.002",
        facecolor=color, edgecolor=color, linewidth=0.5,
        transform=fig.transFigure, zorder=8, clip_on=False))
    sw = bw * 0.58
    fig.patches.append(Arc(
        (x, y + bh / 2), sw, sw * ASP, angle=0, theta1=0, theta2=180,
        color=color, lw=1.1, transform=fig.transFigure, zorder=8,
        clip_on=False))


def draw_network_block(fig, cx, cy, name, sub, color, trainable=True, note=None):
    """Solid colored network block at center (cx, cy) using TEACHER_RECT-style
    geometry. trainable -> solid + nabla tag; frozen -> 50% alpha + padlock.
    `note` adds a small third line inside the block (e.g. student metadata)."""
    _, dy, w, h = TEACHER_RECT
    x, y = cx - w / 2, cy + dy
    alpha = 1.0 if trainable else 0.5
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor=color, edgecolor=COLOR_BOX_EDGE,
        linewidth=1.0, alpha=alpha, zorder=3, clip_on=False))
    tcol = "white" if trainable else "#2a2a2a"
    name_dy, tag_dy = (0.024, 0.004) if note else (0.016, -0.012)
    fig.text(cx, cy + name_dy, name, ha="center", va="center", fontsize=10,
             fontweight="bold", color=tcol, zorder=6)
    tag = sub + ("   trainable $\\nabla$" if trainable else "   frozen")
    fig.text(cx, cy + tag_dy, tag, ha="center", va="center", fontsize=7.0,
             color=tcol, zorder=6)
    if note:
        fig.text(cx, cy - 0.022, note, ha="center", va="center", fontsize=6.0,
                 color=tcol, style="italic", zorder=6)
    if not trainable:
        draw_padlock(fig, x + w - 0.018, y + h - 0.020 * ASP, tcol)


def _block_geom(cx, cy):
    """Return (x, y, w, h) for a network block centered at (cx, cy)."""
    _, dy, w, h = TEACHER_RECT
    return cx - w / 2, cy + dy, w, h


def draw_loss_pill(fig, cx, cy, name, subtitle, w=0.150, h=0.040):
    fig.patches.append(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.018",
        transform=fig.transFigure, facecolor=COLOR_LOSS, edgecolor="none",
        zorder=5, clip_on=False))
    fig.text(cx, cy, name, ha="center", va="center", fontsize=9.5,
             color="white", fontweight="bold", zorder=6)
    fig.text(cx, cy - h / 2 - 0.009, subtitle, ha="center", va="top",
             fontsize=6.4, color="#555", zorder=6)


def _ramp(diverging):
    g = np.linspace(-1, 1, 40) if diverging else np.linspace(0, 1, 40)
    return np.tile(g, (40, 1)) + 0.12 * np.linspace(-1, 1, 40)[:, None]


def draw_output_tile_strip(fig, centers, y, items, tw=0.024, fs=7):
    """Row of small framed tiles with mathtext symbol titles. Returns
    dict label->(x, y) tile center."""
    th = tw * ASP
    pos = {}
    for cx, (key, label) in zip(centers, items):
        ax = fig.add_axes([cx - tw / 2, y - th / 2, tw, th], zorder=3)
        ax.imshow(_ramp(key in TILE_DIVERGING), cmap=TILE_CMAP[key],
                  origin="upper", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_edgecolor("#888")
        ax.set_title(label, fontsize=fs, pad=2)
        pos[label] = (cx, y)
    return pos


def draw_sonde_glyph(fig, x, y, color, color_label="#555"):
    """Radiosonde: balloon circle + solid trailing string with profile ticks."""
    d = 0.014
    fig.patches.append(Ellipse(
        (x, y), d, d * ASP, facecolor="white", edgecolor=color, linewidth=1.3,
        transform=fig.transFigure, zorder=6, clip_on=False))
    y0 = y - d * ASP / 2
    fig.add_artist(Line2D([x, x], [y0, y0 - 0.052], transform=fig.transFigure,
                          color=color, lw=1.2, ls="-", zorder=6))
    for yi in (y0 - 0.014, y0 - 0.029, y0 - 0.044):
        fig.add_artist(Line2D([x - 0.006, x + 0.006], [yi, yi],
                              transform=fig.transFigure, color=color, lw=1.0,
                              zorder=6))
    fig.text(x, y + d * ASP / 2 + 0.005, "radiosonde", ha="center",
             va="bottom", fontsize=6.4, color=color_label, zorder=6)


def draw_mini_label_block(fig, rect, label, color, locked=True, fs=7.2):
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.008",
        transform=fig.transFigure, facecolor=color, edgecolor=COLOR_BOX_EDGE,
        linewidth=0.8, alpha=0.5, zorder=3, clip_on=False))
    fig.text(x + w / 2, y + h / 2, label, ha="center", va="center",
             fontsize=fs, color="#222", zorder=6)
    if locked:
        draw_padlock(fig, x + w - 0.013, y + h - 0.016 * ASP, "#444", w=0.010)


def draw_cylinder(fig, x, y, cw, bh, fill=COLOR_DATA, edge="#555", platters=2,
                  z=3):
    """A simple flat database/disk cylinder centered at (x, y)."""
    eh = cw * 0.30 * ASP
    fig.patches.append(Ellipse((x, y - bh / 2), cw, eh, facecolor=fill,
                               edgecolor=edge, lw=0.9, transform=fig.transFigure,
                               zorder=z, clip_on=False))
    fig.patches.append(Rectangle((x - cw / 2, y - bh / 2), cw, bh,
                                 facecolor=fill, edgecolor="none",
                                 transform=fig.transFigure, zorder=z,
                                 clip_on=False))
    for sx in (x - cw / 2, x + cw / 2):
        fig.add_artist(Line2D([sx, sx], [y - bh / 2, y + bh / 2],
                              transform=fig.transFigure, color=edge, lw=0.9,
                              zorder=z))
    fig.patches.append(Ellipse((x, y + bh / 2), cw, eh, facecolor=fill,
                               edgecolor=edge, lw=0.9, transform=fig.transFigure,
                               zorder=z + 1, clip_on=False))
    for i in range(platters):
        yy = y + bh / 2 - (i + 1) * bh / (platters + 1)
        fig.patches.append(Arc((x, yy), cw, eh, angle=0, theta1=180, theta2=360,
                               color=edge, lw=0.6, transform=fig.transFigure,
                               zorder=z + 1, clip_on=False))


def draw_storage_block(fig, x, y, label, sublabel):
    """Disk-stack storage glyph (tall, narrow) + labels below."""
    cw, bh = 0.032, 0.052
    draw_cylinder(fig, x, y, cw, bh, platters=3)
    base = y - bh / 2 - cw * 0.30 * ASP / 2
    fig.text(x, base - 0.010, label, ha="center", va="top", fontsize=7.6,
             fontweight="bold", color="#333", zorder=6)
    fig.text(x, base - 0.030, sublabel, ha="center", va="top", fontsize=6.4,
             color="#666", zorder=6)


def draw_corpus_glyph(fig, x, y, label, sublabel):
    """Dataset/corpus glyph (wide, short cylinder) + labels below."""
    cw, bh = 0.056, 0.036
    draw_cylinder(fig, x, y, cw, bh, platters=2)
    base = y - bh / 2 - cw * 0.30 * ASP / 2
    fig.text(x, base - 0.010, label, ha="center", va="top", fontsize=7.6,
             fontweight="bold", color="#333", zorder=6)
    fig.text(x, base - 0.030, sublabel, ha="center", va="top", fontsize=6.4,
             color="#666", zorder=6)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(out_base: Path):
    plt.style.use(str(REPO / "figures" / "paper.mplstyle"))
    fig = plt.figure(figsize=(FIGW, FIGH))

    fig.text(0.5, 0.985, "Three-stage training pipeline", ha="center",
             va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.963,
             "(a) Teacher fine-tuning   ·   (b) Offline label generation   ·   "
             "(c) Student distillation", ha="center", va="top", fontsize=9,
             color="#444")

    pill_cx = 0.80                       # loss-pill column center
    pin = pill_cx - 0.078                # where supervision arrows enter a pill
    teach_cx = 0.200                     # network-block center x (bands a & b)
    W = TEACHER_RECT[2]
    blk_l, blk_r = teach_cx - W / 2, teach_cx + W / 2          # 0.151, 0.249
    tile_l = TILE_CENTERS[0] - 0.024 / 2                       # ~0.298
    tile_r = TILE_CENTERS[-1] + 0.024 / 2                      # ~0.446
    tw, th = 0.020, 0.020 * ASP          # input scene-thumbnail size

    # =====================================================================
    # Band (a): teacher fine-tuning
    # =====================================================================
    _stage_box(fig, (0.015, 0.655, 0.97, 0.290), "(a) Teacher fine-tuning", lw=0.6)
    yA = 0.800

    for x, lab in zip([0.040, 0.070, 0.100], ["A$^-$", "A$_0$", "A$^+$"]):
        ax = fig.add_axes([x, yA + 0.010, tw, th], zorder=3)
        ax.imshow(_ramp(False), cmap="gray_r", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_edgecolor(COLOR_A)
        ax.set_title(lab, fontsize=6.2, pad=1, color=COLOR_A)
    for x, lab in zip([0.055, 0.085], ["B$^-$", "B$^+$"]):
        ax = fig.add_axes([x, yA - 0.010 - th, tw, th], zorder=3)
        ax.imshow(_ramp(False), cmap="gray_r", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_edgecolor(COLOR_B)
        ax.text(0.5, -0.22, lab, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color=COLOR_B)
    fig.text(0.070, yA - 0.010 - th - 0.020, "GOES-16 × GOES-18\n(5 scenes)",
             ha="center", va="top", fontsize=6.4, color="#333")

    _arrow(fig, (0.122, yA), (blk_l - 0.003, yA))
    draw_network_block(fig, teach_cx, yA, "Teacher", r"$\theta$", COLOR_A,
                       trainable=True)
    _arrow(fig, (blk_r + 0.003, yA), (tile_l - 0.004, yA))
    tpos = draw_output_tile_strip(fig, TILE_CENTERS, yA, TILE_ITEMS)

    rows = [0.862, 0.788, 0.714]
    draw_loss_pill(fig, pill_cx, rows[0], r"$\mathcal{L}_{\chi^2}$",
                   "self-supervised geometric residual")
    draw_loss_pill(fig, pill_cx, rows[1], r"$\mathcal{L}_{\mathrm{sonde}}$",
                   "Huber on $u,v$ (IGRA)")
    draw_loss_pill(fig, pill_cx, rows[2], r"$\mathcal{L}_{\mathrm{KL}}$",
                   r"height anchor to frozen $\theta_0$")

    # chi2 self-supervised loop: leaves chi2 tile, arcs above the tile row.
    _arrow(fig, (tpos[r"$\chi^2$"][0], yA + th / 2 + 0.004),
           (pin, rows[0]), rad=-0.60, lw=1.3)
    # sonde supervision
    draw_sonde_glyph(fig, 0.660, rows[1] + 0.006, COLOR_LOSS)
    _arrow(fig, (0.678, rows[1]), (pin, rows[1]))
    # KL anchor from frozen theta0
    draw_mini_label_block(fig, (0.534, rows[2] - 0.020, 0.095, 0.040),
                          r"$\theta_0$ pretrained", COLOR_FROZEN, fs=6.4)
    _arrow(fig, (0.633, rows[2]), (pin, rows[2]))

    fig.text(0.975, 0.666,
             r"$\mathcal{L}_{\mathrm{teacher}} = \mathcal{L}_{\chi^2}"
             r" + \lambda_{\mathrm{sonde}}\mathcal{L}_{\mathrm{sonde}}"
             r" + \lambda_h\mathcal{L}_{\mathrm{KL}}$",
             ha="right", va="center", fontsize=8.5, color="#111")

    # connector (a) -> (b)
    _down_arrow(fig, teach_cx, 0.653, 0.607, label="freeze weights")

    # =====================================================================
    # Band (b): offline label generation
    # =====================================================================
    _stage_box(fig, (0.015, 0.415, 0.97, 0.190), "(b) Label generation", lw=0.6)
    yB = 0.532
    th_t = 0.024 * ASP   # output-tile height

    draw_corpus_glyph(fig, 0.070, yB, "Label corpus",
                      "all matched GEO–GEO\nscenes [N=TBD]")
    _arrow(fig, (0.122, yB), (blk_l - 0.003, yB))
    draw_network_block(fig, teach_cx, yB, "Teacher", r"$\theta^*$",
                       COLOR_A, trainable=False)
    fig.text(teach_cx, yB - 0.082, "= same model as (a)", ha="center",
             va="top", fontsize=6.0, style="italic", color="#666", zorder=6)
    _arrow(fig, (blk_r + 0.003, yB), (tile_l - 0.004, yB))
    bpos = draw_output_tile_strip(fig, TILE_CENTERS, yB, TILE_ITEMS)

    # storage at far right; main data-flow arrow outputs -> storage
    stor_x = 0.80
    draw_storage_block(fig, stor_x, yB, "Stored labels", "cached to disk")
    _arrow(fig, (tile_r + 0.004, yB), (stor_x - 0.034, yB))

    # QA-weight derivation block just below the strip (dashed derivation arrows)
    qa = (0.490, 0.470, 0.125, 0.038)
    fig.patches.append(FancyBboxPatch(
        (qa[0], qa[1]), qa[2], qa[3],
        boxstyle="round,pad=0.003,rounding_size=0.008",
        transform=fig.transFigure, facecolor="white", edgecolor="#888",
        linewidth=0.8, zorder=3, clip_on=False))
    fig.text(qa[0] + qa[2] / 2, qa[1] + qa[3] / 2, "QA weight", ha="center",
             va="center", fontsize=7.0, fontweight="bold", color="#333",
             zorder=6)
    fig.text(qa[0] + qa[2] / 2, qa[1] - 0.006,
             r"$\mathrm{QA} = (\chi^2 < \tau_\chi)\ \wedge\ (\sigma_h < \tau_\sigma)$",
             ha="center", va="top", fontsize=6.2, color="#555", zorder=6)
    _arrow(fig, (bpos[r"$\chi^2$"][0], yB - th_t / 2 - 0.002),
           (qa[0] + qa[2] - 0.02, qa[1] + qa[3]), lw=0.9, rad=0.25,
           ls=(0, (3, 2)), color="#777")
    _arrow(fig, (bpos[r"$\sigma_{h,u,v}$"][0], yB - th_t / 2 - 0.002),
           (qa[0] + 0.022, qa[1] + qa[3]), lw=0.9, rad=-0.2,
           ls=(0, (3, 2)), color="#777")
    # QA block -> storage
    _arrow(fig, (qa[0] + qa[2], qa[1] + qa[3] / 2), (stor_x - 0.020, yB - 0.022),
           lw=1.1, rad=-0.2)

    # caption anchored bottom-right (rhythm with the loss equations in a & c)
    fig.text(0.975, 0.422,
             "single offline pass over the corpus;\noutputs cached for reuse",
             ha="right", va="bottom", fontsize=6.4, style="italic",
             color="#666")

    # =====================================================================
    # Band (c): student distillation
    # =====================================================================
    _stage_box(fig, (0.015, 0.045, 0.97, 0.300), "(c) Student distillation", lw=0.6)
    yC = 0.200

    stw, sth = 0.028, 0.028 * ASP
    ax = fig.add_axes([0.050, yC - sth / 2, stw, sth], zorder=3)
    ax.imshow(_ramp(False), cmap="gray_r", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7); s.set_edgecolor(COLOR_STUDENT)
    ax.set_title("Single GEO", fontsize=6.6, pad=2, color=COLOR_STUDENT)
    fig.text(0.064, yC - sth / 2 - 0.009, "one viewpoint\nmulti-band stack",
             ha="center", va="top", fontsize=6.2, color="#333")

    _arrow(fig, (0.122, yC), (blk_l - 0.003, yC))
    draw_network_block(fig, teach_cx, yC, "Student", r"$\varphi$",
                       COLOR_STUDENT, trainable=True,
                       note="predicts own $\\sigma$ (NLL)")
    _arrow(fig, (blk_r + 0.003, yC), (tile_l - 0.004, yC))
    spos = draw_output_tile_strip(fig, TILE_CENTERS, yC, TILE_ITEMS)

    s_rows = [0.300, 0.220, 0.140]
    draw_loss_pill(fig, pill_cx, s_rows[0], r"$\mathcal{L}^{\mathrm{NLL}}_{uv}$",
                   "bivariate Gaussian NLL on $(u,v)$")
    draw_loss_pill(fig, pill_cx, s_rows[1], r"$\mathcal{L}^{\mathrm{NLL}}_{h}$",
                   "Gaussian NLL on $h$")
    draw_loss_pill(fig, pill_cx, s_rows[2],
                   r"$\mathcal{L}_{\chi^2\text{-}\mathrm{distill}}$",
                   r"matches teacher $\chi^2$ (L1 on $\log\chi^2$)")

    # ---- loaded-labels landing pad (same row as the student outputs) ---
    pad = (0.466, yC - 0.022, 0.105, 0.044)
    fig.patches.append(FancyBboxPatch(
        (pad[0], pad[1]), pad[2], pad[3],
        boxstyle="round,pad=0.003,rounding_size=0.008",
        transform=fig.transFigure, facecolor="white", edgecolor="#888",
        linewidth=0.8, zorder=2, clip_on=False))
    fig.text(pad[0] + pad[2] / 2, pad[1] + pad[3] + 0.006, "labels from (b)",
             ha="center", va="bottom", fontsize=7.5, style="italic",
             color="#555", zorder=6)
    lab_items = [("u", "$u$"), ("v", "$v$"), ("h", "$h$"), ("chi2", r"$\chi^2$")]
    lab_centers = [pad[0] + 0.019 + i * 0.022 for i in range(4)]
    draw_output_tile_strip(fig, lab_centers, pad[1] + pad[3] / 2 - 0.002,
                           lab_items, tw=0.013, fs=5.5)

    # connector (b) -> (c): stored labels drop into the landing pad (straight),
    # label set parallel to the arrow.
    _arrow(fig, (stor_x + 0.010, 0.452), (pad[0] + pad[2] * 0.5, pad[1] + pad[3]),
           lw=1.4, rad=0.0)
    fig.text(0.668, 0.342, "load stored labels", rotation=29,
             rotation_mode="anchor", ha="center", va="bottom", fontsize=7.0,
             style="italic", color=COLOR_ARROW, zorder=6)

    # labels fan from the pad up to each loss pill (neutral gray group)
    lab_src = (pad[0] + pad[2], pad[1] + pad[3] / 2)
    for sr in s_rows:
        _arrow(fig, lab_src, (pin, sr), lw=1.2, rad=0.0, color="#777")

    fig.text(0.975, 0.082,
             r"$\mathcal{L}_{\mathrm{student}} ="
             r" \mathcal{L}^{\mathrm{NLL}}_{uv}"
             r" + \lambda_h\,\mathcal{L}^{\mathrm{NLL}}_{h}"
             r" + \lambda_{\chi^2}\,\mathcal{L}_{\chi^2}$",
             ha="right", va="center", fontsize=8.5, color="#111")
    fig.text(0.975, 0.058, "all terms masked + weighted by QA from (b)",
             ha="right", va="center", fontsize=6.0, style="italic",
             color="#666")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        logger.info("Wrote %s", path)
    plt.close(fig)


if __name__ == "__main__":
    build_figure(REPO / "figures" / "fig_training_overview")
