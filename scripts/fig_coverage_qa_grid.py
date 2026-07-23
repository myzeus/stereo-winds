#!/usr/bin/env python
"""Full-disk wind coverage vs. QA level: a 2-row x N-column grid of GOES-19
geostationary barb maps, barbs colored by retrieved cloud-top height.

Top row = cross-satellite stereo (teacher); bottom row = single-satellite
student.  Columns tighten the chi-squared QA gate (No QA -> chi2<=0.3), so the
figure shows how retrieval coverage shrinks as quality control tightens, for
both retrievals side by side.

Runs on ADAPT against the full-disk zarrs, or locally via --from-bundle:
    python scripts/fig_coverage_qa_grid.py \
        --teacher-zarr hreg1s75_C14_202510_iter3.zarr \
        --student-zarr student_quad_ab_C14.zarr \
        --time 2025-10-01T12:00 --dump-bundle cov.npz --out figures/fig_coverage_qa.png
    python scripts/fig_coverage_qa_grid.py --from-bundle cov.npz --out figures/fig_coverage_qa.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

H_MIN, H_MAX = 0.0, 16000.0
MS_TO_KT = 1.94384
# (label, chi2_max or None for no-QA)
QA_LEVELS = [("No QA", None), (r"$\chi^2\!\leq\!1.0$", 1.0),
             (r"$\chi^2\!\leq\!0.5$", 0.5), (r"$\chi^2\!\leq\!0.3$", 0.3)]
ROWS = [("(a) Cross-satellite stereo (teacher)", "teacher"),
        ("(b) Single-satellite student", "student")]


def qa_mask(u, v, h, chi2, sigh, qf, chi2_max):
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(h)
    if chi2_max is None:
        return finite
    spd = np.hypot(u, v)
    return (finite & np.isfinite(chi2) & (qf > 0) & (chi2 <= chi2_max)
            & np.isfinite(sigh) & (sigh <= 5000.0)
            & (spd <= 120.0) & (h >= H_MIN) & (h <= 20000.0))


def thinned_barbs(ds_t, lon, lat, chi2_max, stride):
    """Return (lon, lat, u_kt, v_kt, h_km) for QA-passing pixels on a strided grid."""
    u = ds_t["u_wind"].values; v = ds_t["v_wind"].values
    h = ds_t["cloud_top_height"].values
    m = qa_mask(u, v, h, ds_t["chi_squared"].values, ds_t["sigma_h"].values,
                ds_t["quality_flag"].values, chi2_max)
    keep = np.zeros_like(m)
    keep[::stride, ::stride] = True
    mm = m & keep
    return (lon[mm], lat[mm], u[mm] * MS_TO_KT, v[mm] * MS_TO_KT, h[mm] / 1000.0,
            int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-zarr")
    ap.add_argument("--student-zarr")
    ap.add_argument("--time", default="2025-10-01T12:00")
    ap.add_argument("--stride", type=int, default=58)
    ap.add_argument("--band", default="C14")
    ap.add_argument("--bt-zarr", default=None,
                    help="radiance/BT cube for the greyscale underlay (same scene)")
    ap.add_argument("--bt-stride", type=int, default=4,
                    help="coarsen the BT underlay by this factor before storing/plotting")
    ap.add_argument("--bt-alpha", type=float, default=0.85,
                    help="opacity of the BT underlay")
    ap.add_argument("--bt-lo", type=float, default=0.10,
                    help="lightest grey (fraction into Greys); >0 keeps cold cloud off pure white")
    ap.add_argument("--bt-hi", type=float, default=0.60,
                    help="darkest grey (fraction into Greys); <1 keeps warm air off pure black")
    ap.add_argument("--no-barb-halo", action="store_true",
                    help="disable the white outline stroke on barbs")
    ap.add_argument("--dump-bundle", default=None)
    ap.add_argument("--from-bundle", default=None)
    ap.add_argument("--out", default="figures/fig_coverage_qa.png")
    args = ap.parse_args()

    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent
    style = REPO / "figures" / "paper.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    # panels[row][col] = (lon, lat, ukt, vkt, hkm, n_total); + tstr
    bt = None; bt_extent = None
    if args.from_bundle:
        b = np.load(args.from_bundle, allow_pickle=True)
        panels = b["panels"].tolist()
        tstr = str(b["tstr"]); ntot = b["ntot"].tolist()
        if "bt" in b.files and b["bt"].ndim == 2:
            bt = b["bt"]; bt_extent = b["bt_extent"].tolist()
    else:
        import xarray as xr
        from stereo_winds.config import GOES19_CONFIG
        from stereo_winds.navigation import fixed_grid_to_geodetic
        sat = GOES19_CONFIG
        cols = np.arange(sat.n_cols); rows = np.arange(sat.n_rows)
        xr_ = cols * sat.scale_x + sat.x_offset
        yr_ = rows * sat.scale_y + sat.y_offset
        xg, yg = np.meshgrid(xr_, yr_)
        lat, lon = fixed_grid_to_geodetic(xg, yg, sat)

        if args.bt_zarr:
            dbt = xr.open_zarr(args.bt_zarr)
            if "time" in dbt.dims:
                dbt = dbt.sel(time=np.datetime64(args.time), method="nearest")
            f = args.bt_stride
            bt = dbt["Rad"].values[::f, ::f].astype(np.float32)
            H = sat.satellite_height_m
            bt_extent = [float(xr_[0] * H), float(xr_[-1] * H),
                         float(yr_[-1] * H), float(yr_[0] * H)]
        srcs = {"teacher": args.teacher_zarr, "student": args.student_zarr}
        opened = {}
        tstr = None
        for k, p in srcs.items():
            ds = xr.open_zarr(p)
            if "band" in ds.dims:
                ds = ds.sel(band=args.band) if args.band in list(ds.band.values) else ds.isel(band=-1)
            if "time" in ds.dims:
                ds = ds.sel(time=np.datetime64(args.time), method="nearest")
            ds.load()
            opened[k] = ds
            tstr = str(ds.time.values)[:16] if "time" in ds.coords else args.time
        panels = []; ntot = []
        for _, key in ROWS:
            rowp = []; rown = []
            for _, cm in QA_LEVELS:
                lo, la, uk, vk, hk, n = thinned_barbs(opened[key], lon, lat, cm, args.stride)
                rowp.append((lo, la, uk, vk, hk)); rown.append(n)
            panels.append(rowp); ntot.append(rown)
        if args.dump_bundle:
            extra = {}
            if bt is not None:
                extra = dict(bt=bt, bt_extent=np.array(bt_extent))
            np.savez_compressed(args.dump_bundle,
                                panels=np.array(panels, dtype=object),
                                ntot=np.array(ntot), tstr=tstr, **extra)
            print(f"=== dumped bundle {args.dump_bundle}")

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from stereo_winds.config import GOES19_CONFIG
    # mid-light grey band for the underlay: never pure white / black so every
    # height-coloured barb keeps contrast against it.
    bt_cmap = LinearSegmentedColormap.from_list(
        "lightIR", plt.get_cmap("Greys")(np.linspace(args.bt_lo, args.bt_hi, 256)))
    halo = [] if args.no_barb_halo else [pe.withStroke(linewidth=1.1, foreground="white")]
    sat = GOES19_CONFIG
    PC = ccrs.PlateCarree()
    geo = ccrs.Geostationary(central_longitude=sat.sub_lon_deg,
                             satellite_height=sat.satellite_height_m,
                             sweep_axis=sat.sweep)
    norm = Normalize(vmin=H_MIN / 1000.0, vmax=H_MAX / 1000.0)
    cmap = plt.get_cmap("cividis")

    nrow, ncol = len(ROWS), len(QA_LEVELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.35 * nrow),
                             subplot_kw={"projection": geo})
    for i, (rlab, _) in enumerate(ROWS):
        for j, (clab, _) in enumerate(QA_LEVELS):
            ax = axes[i, j]
            ax.set_global()
            if bt is not None:
                bvmin, bvmax = np.nanpercentile(bt[np.isfinite(bt)], [2, 98])
                ax.imshow(bt, origin="upper", extent=bt_extent, transform=geo,
                          cmap=bt_cmap, vmin=bvmin, vmax=bvmax, alpha=args.bt_alpha,
                          zorder=0, interpolation="bilinear")
            ax.coastlines(resolution="110m", color="0.45", linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, edgecolor="0.65", linewidth=0.15)
            lo, la, uk, vk, hk = panels[i][j]
            if len(lo):
                bb = ax.barbs(lo, la, uk, vk, hk, cmap=cmap, norm=norm, transform=PC,
                              length=3.4, linewidth=0.45, zorder=3)
                if halo:
                    bb.set_path_effects(halo)
            ax.text(0.5, -0.02, f"{ntot[i][j]:,} px", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color="0.3")
            if i == 0:
                ax.set_title(clab, fontsize=11, fontweight="bold")
            if j == 0:
                ax.text(-0.06, 0.5, rlab, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, fontweight="bold")

    fig.suptitle(f"Full-disk retrieval coverage vs. QA threshold  ·  {args.band}  ·  "
                 f"{tstr} UTC", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.045, right=0.90, top=0.90, bottom=0.05,
                        wspace=0.04, hspace=0.08)
    cax = fig.add_axes([0.915, 0.12, 0.013, 0.72])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("Retrieved cloud-top height (km)")

    for ext in ("png", "pdf"):
        p = Path(args.out).with_suffix(f".{ext}")
        fig.savefig(p, dpi=200)
        print(f"=== wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
